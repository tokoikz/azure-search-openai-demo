import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "うれしい、たのしい、大好き",
        value: "うれしい、たのしい、大好き"
    },
    { text: "悲しい、切ない、物憂げな", value: "悲しい、切ない、物憂げな" },
    { text: "楽しい、おかしい、弾むような", value: "楽しい、おかしい、弾むような" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList2 = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
