export class SGDTrainer {
    constructor(model, learningRate) {
        this.model = model;
        this.learningRate = learningRate;
    }

    // Mean cross-entropy over the unmasked positions.
    // A mask value of 0 marks a context-only token (skipped), 1 marks a token
    // that should contribute to the loss. With no mask, every position counts.
    crossEntropyLoss(logits, targetIds, targetMask = null) {
        // logits:     Matrix(sequenceLength, vocabSize) — Values
        // targetIds:  Array(sequenceLength)             — plain integers
        // targetMask: Array(sequenceLength), or null
        const probs = logits.softMaxedRows();
        const sequenceLength = logits.rows;

        let loss = null;
        let count = 0;

        for (let i = 0; i < sequenceLength; ++i) {
            // A masked-out position is context only: no loss, no gradient.
            if (targetMask !== null && !targetMask[i]) {
                continue;
            }

            const targetProb = probs.get(i, targetIds[i]);
            const tokenLoss = targetProb.log().neg();

            if (loss === null) {
                loss = tokenLoss;
            }
            else {
                loss = loss.add(tokenLoss);
            }

            count += 1;
        }

        if (count === 0) {
            throw new Error("No unmasked targets were provided");
        }

        // Average only over the positions that were counted.
        return loss.mul(1 / count);
    }

    train(tokenIds, tokenMask = null) { // tokenMask: Array(N) of 0/1, or null
        const N = tokenIds.length;
        if (N < 2) {
            throw new Error(`need at least 2 tokens, got ${N}`);
        }

        if (tokenMask !== null && tokenMask.length !== N) {
            throw new Error(`mask length ${tokenMask.length} does not match token length ${N}`);
        }

        const inputIds = tokenIds.slice(0, N - 1); // fed to the model
        const targetIds = tokenIds.slice(1);       // the same sequence shifted by 1
        // The mask lines up with tokenIds, so shift it like the targets. The
        // first token is never a target, so its mask is dropped.
        const targetMask = tokenMask === null ? null : tokenMask.slice(1);

        // Clear gradients.
        const params = this.model.parameters();
        for (let i = 0; i < params.length; ++i) {
            params[i].partial = 0;
        }

        // Forward -> loss -> backward.
        const logits = this.model.forward(inputIds);                       // Matrix(N-1, vocabSize)
        const loss = this.crossEntropyLoss(logits, targetIds, targetMask); // single scalar Value
        loss.backward();                                                   // fills in every partial

        // Step.
        // Mutating .data directly bypasses the graph
        for (let i = 0; i < params.length; ++i) {
            params[i].data -= this.learningRate * params[i].partial;
        }

        return loss.data; // plain number, for logging
    }
}
