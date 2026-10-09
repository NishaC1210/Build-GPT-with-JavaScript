
export class SGDTrainer {
    constructor(model, learningRate) {
        this.model = model;
        this.learningRate = learningRate;
    }

    // Mean cross-entropy: -1/N * sum_i log(q[i, targetIds[i]])
    // where q = softmax(logits), one row per prediction.
    crossEntropyLoss(logits, targetIds) {
        // logits:    Matrix(sequenceLength, vocabSize) — Values
        // targetIds: Array(sequenceLength)             — plain integers
        const probs = logits.softMaxedRows();
        const sequenceLength = logits.rows;

        // Seed with row 0, then chain .add — the same pattern as matmul/softmax.
        // row i -> position in the sequence; col targetIds[i] -> the next token id.
        let loss = probs.get(0, targetIds[0]).log().neg();
        for (let i = 1; i < sequenceLength; ++i) {
            const targetProb = probs.get(i, targetIds[i]);
            loss = loss.add(targetProb.log().neg());
        }
        return loss.mul(1 / sequenceLength);
    }

    train(tokenIds) { // tokenIds: Array(N) of ints,  one training sequence
        const N = tokenIds.length;
        if (N < 2) throw new Error(`need at least 2 tokens, got ${N}`);

        const inputIds = tokenIds.slice(0, N - 1); // fed to the model
        const targetIds = tokenIds.slice(1);       // the same sequence shifted by 1

        // Clear gradients.
        const params = this.model.parameters();
        for (let i = 0; i < params.length; ++i) {
            params[i].partial = 0;
        }

        // Forward -> loss -> backward.
        const logits = this.model.forward(inputIds);           // Matrix(N-1, vocabSize)
        const loss = this.crossEntropyLoss(logits, targetIds); // single scalar Value
        loss.backward();                                       // fills in every partial

        // Step. 
        for (let i = 0; i < params.length; ++i) {
            params[i].data -= this.learningRate * params[i].partial;
        }

        return loss.data; // plain number, for logging
    }
}