import { Tensor } from './Tensor.js';

export class AdamWTrainer {
    constructor(model, {
        maxLearningRate = 3e-4,
        minLearningRate = 3e-5,
        warmupSteps = 100,
        totalSteps = 10000,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
        weightDecay = 0.01,
        gradientClip = 1.0,
    } = {}) {
        this.model = model;

        this.maxLearningRate = maxLearningRate;
        this.minLearningRate = minLearningRate;
        this.warmupSteps = warmupSteps;
        this.totalSteps = totalSteps;

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.gradientClip = gradientClip;

        // Grab the parameter list once. The tensors inside are stable references,
        // so the moment buffers below stay matched to them for the whole run.
        this.params = model.parameters();

        // Adam keeps two running averages per parameter, each the same shape as
        // the parameter itself. A fresh Float32Array is already all zeros.
        this.firstMoments = [];
        this.secondMoments = [];
        for (let i = 0; i < this.params.length; ++i) {
            const length = this.params[i].data.length;
            this.firstMoments.push(new Float32Array(length));  // m
            this.secondMoments.push(new Float32Array(length)); // v
        }

        // Updates applied so far. Drives both bias correction and the schedule.
        this.step = 0;

        // The learning rate actually used on the last step, exposed for logging.
        this.lastLearningRate = 0;
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

    learningRate() {
        const step = this.step;

        // Warmup: ramp linearly from zero to the peak over the first steps.
        if (step < this.warmupSteps) {
            return this.maxLearningRate * (step / this.warmupSteps);
        }

        // Past the planned end: hold at the floor.
        if (step >= this.totalSteps) {
            return this.minLearningRate;
        }

        // Cosine decay: ease from the peak down to the floor.
        const progress = (step - this.warmupSteps) / (this.totalSteps - this.warmupSteps);
        const cosine = 0.5 * (1 + Math.cos(Math.PI * progress));
        return this.minLearningRate + (this.maxLearningRate - this.minLearningRate) * cosine;
    }

    clipGradients() {
        // Global L2 norm across every gradient in the model.
        let sumSquares = 0.0;
        for (let i = 0; i < this.params.length; ++i) {
            const grad = this.params[i].grad;
            for (let j = 0; j < grad.length; ++j) {
                sumSquares += grad[j] * grad[j];
            }
        }
        const norm = Math.sqrt(sumSquares);

        // Only scale down when the norm is over the threshold. A smaller
        // update is left alone.
        if (norm > this.gradientClip) {
            const scale = this.gradientClip / norm;
            for (let i = 0; i < this.params.length; ++i) {
                const grad = this.params[i].grad;
                for (let j = 0; j < grad.length; ++j) {
                    grad[j] *= scale;
                }
            }
        }

        return norm; // useful for logging
    }

    train(tokenIdBatch, tokenMaskBatch = null) {
        if (!Array.isArray(tokenIdBatch) || tokenIdBatch.length === 0) {
            throw new Error("Mini-batch must contain at least one training example");
        }
        if (tokenMaskBatch !== null && tokenMaskBatch.length !== tokenIdBatch.length) {
            throw new Error(
                `mask batch length ${tokenMaskBatch.length} does not match ` +
                `token batch length ${tokenIdBatch.length}`
            );
        }

        // Clear gradients once, then let every training example add to them.
        for (let i = 0; i < this.params.length; ++i) {
            this.params[i].zeroGrad();
        }

        let totalLoss = 0.0;
        for (let batchIndex = 0; batchIndex < tokenIdBatch.length; ++batchIndex) {
            const tokenIds = tokenIdBatch[batchIndex];
            const tokenMask = tokenMaskBatch === null ? null : tokenMaskBatch[batchIndex];

            if (tokenIds.length < 2) {
                throw new Error(
                    `training example ${batchIndex} needs at least 2 tokens, ` +
                    `got ${tokenIds.length}`
                );
            }
            if (tokenMask !== null && tokenMask.length !== tokenIds.length) {
                throw new Error(
                    `mask length ${tokenMask.length} does not match ` +
                    `token length ${tokenIds.length} for example ${batchIndex}`
                );
            }

            const inputIds  = tokenIds.slice(0, tokenIds.length - 1);
            const targetIds = tokenIds.slice(1);
            // The mask lines up with tokenIds, so shift it like the targets. The
            // first token is never a target, so its mask is dropped.
            const targetMask = tokenMask === null ? null : tokenMask.slice(1);

            const logits = this.model.forward(inputIds);
            const loss = this.crossEntropyLoss(logits, targetIds, targetMask);
            loss.backward();
            totalLoss += loss.data[0];
        }

        // backward() added one gradient per training example. Their mean is the
        // gradient for the mini-batch.
        const batchScale = 1 / tokenIdBatch.length;
        for (let i = 0; i < this.params.length; ++i) {
            const grad = this.params[i].grad;
            for (let j = 0; j < grad.length; ++j) {
                grad[j] *= batchScale;
            }
        }

        // Rein in the averaged gradient before it moves anything.
        this.clipGradients();

        // Advance the step, then read this step's scheduled learning rate.
        this.step += 1;
        const learningRate = this.learningRate();
        this.lastLearningRate = learningRate;

        // Cache hyperparameters as locals for the inner loop.
        const beta1 = this.beta1;
        const beta2 = this.beta2;
        const epsilon = this.epsilon;
        const weightDecay = this.weightDecay;

        // Bias-correction denominators for this step.
        const correction1 = 1 - Math.pow(beta1, this.step);
        const correction2 = 1 - Math.pow(beta2, this.step);

        // AdamW update, parameter by parameter.
        for (let i = 0; i < this.params.length; ++i) {
            const param = this.params[i];
            const m = this.firstMoments[i];
            const v = this.secondMoments[i];

            for (let j = 0; j < param.data.length; ++j) {
                const g = param.grad[j];
                const weight = param.data[j];

                // Update the running averages of the gradient and its square.
                m[j] = beta1 * m[j] + (1 - beta1) * g;
                v[j] = beta2 * v[j] + (1 - beta2) * g * g;

                // Undo the startup bias toward zero.
                const mHat = m[j] / correction1;
                const vHat = v[j] / correction2;

                // The Adam step: smoothed direction, per-parameter scale.
                const adamStep = mHat / (Math.sqrt(vHat) + epsilon);

                // Decoupled weight decay: pull the weight toward zero on its
                // own, not routed through the adaptive scaling above.
                const decayStep = weightDecay * weight;

                param.data[j] = weight - learningRate * (adamStep + decayStep);
            }
        }

        return totalLoss * batchScale;
    }
}
