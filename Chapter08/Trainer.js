import { Tensor } from './Tensor.js';

export class SGDTrainer {
    constructor(model, learningRate) {
        this.model = model;
        this.learningRate = learningRate;
    }

    // Fused mean cross-entropy. The forward pass uses the log-sum-exp form for
    // each row of logits, which avoids exponentiating large values directly:
    // for a row with correct token t, the loss is log(sum_j e^{x_j}) - x_t.
    // The softmax probabilities are computed along the way and cached, because
    // the backward rule reuses them.
    crossEntropyLoss(logits, targetIds, targetMask = null) { // mask: new argument
        const rows = logits.rows;
        const cols = logits.columns;
        const probs = new Float32Array(rows * cols); // cached for the backward
        let total = 0.0;
        let count = 0; // mask: unmasked rows — the divisor for the mean

        for (let row = 0; row < rows; ++row) {
            // mask: a context-only row contributes no loss and no gradient.
            // Skip it entirely; its probs are never read.
            if (targetMask !== null && !targetMask[row]) {
                continue;
            }

            const base = row * cols;

            let rowMax = logits.data[base];
            for (let col = 1; col < cols; ++col) {
                const v = logits.data[base + col];
                if (v > rowMax) rowMax = v;
            }

            let sum = 0.0;
            for (let col = 0; col < cols; ++col) {
                const e = Math.exp(logits.data[base + col] - rowMax);
                probs[base + col] = e;
                sum += e;
            }
            for (let col = 0; col < cols; ++col) {
                probs[base + col] /= sum;
            }

            const logSumExp = rowMax + Math.log(sum);
            total += logSumExp - logits.data[base + targetIds[row]];
            count += 1; // mask: this row counted toward the mean
        }

        if (count === 0) {
            throw new Error("No unmasked targets were provided");
        }

        const loss = new Tensor(1, 1);
        loss.data[0] = total / count; // mask: divide by unmasked count, not rows

        if (Tensor.gradEnabled) {
            loss._inputs = [logits];
            loss._backward = () => {
                const seed = loss.grad[0] / count; // mask: count, not rows
                for (let row = 0; row < rows; ++row) {
                    // mask: skipped rows got no probs, so they get no gradient.
                    if (targetMask !== null && !targetMask[row]) {
                        continue;
                    }
                    const base = row * cols;
                    for (let col = 0; col < cols; ++col) {
                        logits.grad[base + col] += seed * probs[base + col];
                    }
                    logits.grad[base + targetIds[row]] -= seed;
                }
            };
        }
        return loss;
    }

    train(tokenIds, tokenMask = null) { // mask: new argument
        const inputIds  = tokenIds.slice(0, tokenIds.length - 1);
        const targetIds = tokenIds.slice(1);
        // mask: the mask lines up with tokenIds, so shift it like the targets.
        // The first token is never a target, so its mask is dropped.
        const targetMask = tokenMask === null ? null : tokenMask.slice(1);

        const params = this.model.parameters();
        for (const t of params) {
            t.zeroGrad();
        }

        const logits = this.model.forward(inputIds);
        const loss = this.crossEntropyLoss(logits, targetIds, targetMask); // mask: passed through
        loss.backward();

        for (const t of params) {
            for (let i = 0; i < t.data.length; ++i) {
                t.data[i] -= this.learningRate * t.grad[i];
            }
        }

        return loss.data[0];
    }
}
