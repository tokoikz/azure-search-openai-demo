import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.

# Cognitive SearchとOpenAIのAPIを直接使用した、シンプルな retrieve-then-read の実装です。これは、最初に
# 検索からトップ文書を抽出し、それを使ってプロンプトを構成し、OpenAIで補完生成する (answer)をそのプロンプトで表示します。

class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """<|im_start|>system
商品のflavor_score一覧に関する質問をサポートする教師アシスタントです。回答は簡潔にしてください。
今から、flavor_scoreをもつ商品情報を入力しますので、
与えられた商品の一覧情報から、flavor_scoreの最も近い商品を抽出し、
# 出力フォーマットに従って商品名とflavor_scoreを出力してください。
# 次の作業を順次行ってください。
・入力された商品情報からflavor_scoreを取得してください
・下記の商品一覧情報の中で、flavor_scoreの最も近いものを選んでください。
・選んだ商品情報を、以下の出力フォーマットで示してください。
＊ただし、出力するのは、# 出力フォーマットで示されている情報だけで結構です。
途中の作業プロセスは出力に含めないでください。
# 出力フォーマット
商品名：flavor_score
例）日本酒1：10

# 入力商品情報
｛ユーザーが入力｝
{follow_up_questions_prompt}
{injected_prompt}
Sources:
{sources}
<|im_end|>
{chat_history}

"""

    follow_up_questions_prompt_content = """商品について、ユーザーが次に尋ねそうな非常に簡潔なフォローアップ質問を3つ作成する。
    質問を参照するには、二重の角括弧を使用します（例：<<徳川家康とは何をした人ですか?>>）。
    すでに聞かれた質問を繰り返さないようにしましょう。
    質問のみを生成し、「次の質問」のような質問の前後にテキストを生成しない。"""

    query_prompt_template = """以下は、これまでの会話の履歴と、商品情報一覧に関するナレッジベースを検索して回答する必要がある、ユーザーからの新しい質問です。
    会話と新しい質問に基づいて、検索クエリを作成します。
    検索クエリには、引用元のファイル名や文書名（info.txtやdoc.pdfなど）を含めないでください。
    検索キーワードに[]または<<>>内のテキストを含めないでください。

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: list[dict], overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=32, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text
        print(q)

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="ja-jp", 
                                          query_speller="none", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        print(len(prompt),prompt)
        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.0, 
            max_tokens=2048, 
            n=1, 
            stop=["<|im_end|>", "<|im_start|>"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text

class Approach1(ChatReadRetrieveReadApproach):
   prompt_prefix = """<|im_start|>system
商品に関する質問をサポートする教師アシスタントです。回答は簡潔にしてください。
今から、flavor_scoreをもつ商品情報を入力しますので、
与えられた商品の一覧情報から、flavor_scoreの最も近い商品を抽出し、
# 出力フォーマットに従って商品名とflavor_scoreを出力してください。
# 次の作業を順次行ってください。
・入力された商品情報からflavor_scoreを取得してください
・下記の商品一覧情報の中で、flavor_scoreの最も近いものを選んでください。
・選んだ商品情報を、以下の出力フォーマットで示してください。
＊ただし、出力するのは、# 出力フォーマットで示されている情報だけで結構です。
途中の作業プロセスは出力に含めないでください。
# 出力フォーマット
商品名：flavor_score
例）日本酒1：10

# 入力商品情報
｛ユーザーが入力｝
{follow_up_questions_prompt}
{injected_prompt}
Sources:
{sources}
<|im_end|>
{chat_history}

"""

follow_up_questions_prompt_content = """商品について、ユーザーが次に尋ねそうな非常に簡潔なフォローアップ質問を3つ作成する。
    質問を参照するには、二重の角括弧を使用します（例：<<徳川家康とは何をした人ですか?>>）。
    すでに聞かれた質問を繰り返さないようにしましょう。
    質問のみを生成し、「次の質問」のような質問の前後にテキストを生成しない。"""

query_prompt_template = """以下は、これまでの会話の履歴と、商品に関するナレッジベースを検索して回答する必要がある、ユーザーからの新しい質問です。
    会話と新しい質問に基づいて、検索クエリを作成します。
    検索クエリには、引用元のファイル名や文書名（info.txtやdoc.pdfなど）を含めないでください。
    検索キーワードに[]または<<>>内のテキストを含めないでください。

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

class Approach2(ChatReadRetrieveReadApproach):
    prompt_prefix = """<|im_start|>system
あなたは文書を書いたり読んだりするプロです。
以下の作業手順を実行してください。まずはユーザからの感性表現を待ちます。
出力形式は# 出力形式にならって、結果として選ばれた商品情報だけを出力してください。
途中の作業経過は出力不要です。

# 作業
・まずはユーザがカンマ区切りの感性表現を入力します。
・次に感性表現を用いて100文字程度の作文をします。
・100文字程度の文章と、商品情報の商品説明を比較し、最も調子が似ているものを選びます。
・最後に、選ばれた商品情報一覧の中の商品情報を出力してください。

# 出力形式
商品名：
商品説明：
flavor_score
"""
    query_prompt_template = """以下は、これまでの会話の履歴と、商品に関するナレッジベースを検索して回答する必要がある、ユーザーからの新しい質問です。
    会話と新しい質問に基づいて、検索クエリを作成します。
    検索クエリには、引用元のファイル名や文書名（info.txtやdoc.pdfなど）を含めないでください。
    検索キーワードに[]または<<>>内のテキストを含めないでください。

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    follow_up_questions_prompt_content = """商品について、ユーザーが次に尋ねそうな非常に簡潔なフォローアップ質問を3つ作成する。
    質問を参照するには、二重の角括弧を使用します（例：<<徳川家康とは何をした人ですか?>>）。
    すでに聞かれた質問を繰り返さないようにしましょう。
    質問のみを生成し、「次の質問」のような質問の前後にテキストを生成しない。"""

class Approach3(ChatReadRetrieveReadApproach):
    prompt_prefix = """Approach3 specific content here..."""
    query_prompt_template = """Approach3 query template here..."""
