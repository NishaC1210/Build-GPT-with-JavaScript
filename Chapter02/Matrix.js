let randomState = 42;

function random() {
    randomState =
        (1664525 * randomState + 1013904223) % 4294967296;
    return randomState / 4294967296;
}

class Matrix {
    constructor(numRows, numColumns) {
        this.rows = numRows;
        this.columns = numColumns;
        this.values = new Array(this.rows * this.columns);
        // Values are intentionally uninitialized.
        // Call an initialization method before use.
    }

    initToZeroes() {
        for (let i = 0; i < this.values.length; ++i) {
            this.values[i] = 0.0
        }
    }

    initToOnes() {
        for (let i = 0; i < this.values.length; ++i) {
            this.values[i] = 1.0
        }
    }

    initToSmallRandom() {
        const scale = 1 / Math.sqrt(this.rows);
        for (let i = 0; i < this.values.length; ++i) {
            // uniform in [-scale, +scale)
            this.values[i] = (random() - 0.5) * 2 * scale; 
        }
    }

    get(row, column) {
        return this.values[row * this.columns + column];
    }

    set(row, column, value) {
        this.values[row * this.columns + column] = value;
    }

    transposed() {
        const result = new Matrix(this.columns, this.rows);

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                const thisIndex = row * this.columns + col;
                const resultIndex = col * result.columns + row;

                result.values[resultIndex] = this.values[thisIndex];
            }
        }

        return result;
    }

    add(other) {
        const result = new Matrix(this.rows, this.columns);

        for (let i = 0; i < this.values.length; ++i) {
            result.values[i] = this.values[i] + other.values[i];
        }

        return result;

    }

    scale(scalar) {
        const result = new Matrix(this.rows, this.columns);

        for (let i = 0; i < this.values.length; ++i) {
            result.values[i] = this.values[i] * scalar;
        }

        return result;
    }

    matmul(other) {
        const result = new Matrix(this.rows, other.columns);

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < other.columns; ++col) {
                let sum = 0.0;

                for (let element = 0; element < this.columns; ++element) {
                    const a = this.values[row * this.columns + element];
                    const b = other.values[element * other.columns + col];
                    sum += a * b;
                }

                result.values[row * other.columns + col] = sum;
            }
        }

        return result;
    }

    softMaxedRows() {
        const result = new Matrix(this.rows, this.columns);

        for (let row = 0; row < this.rows; ++row) {
            const base = row * this.columns;

            // 1. Find max value in this row (for numerical stability).
            let rowMax = this.values[base];
            for (let col = 1; col < this.columns; ++col) {
                const v = this.values[base + col];
                if (v > rowMax) rowMax = v;
            }

            // 2. Exponentiate each value and sum the row.
            const exps = new Array(this.columns);
            let rowSum = 0.0;
            for (let col = 0; col < this.columns; ++col) {
                const e = Math.exp(this.values[base + col] - rowMax);
                exps[col] = e;
                rowSum += e;
            }

            // 3. Normalize: each output cell = exps[col] / rowSum.
            for (let col = 0; col < this.columns; ++col) {
                result.values[base + col] = exps[col] / rowSum;
            }
        }

        return result;
    }

    causalMasked() {
        const result = new Matrix(this.rows, this.columns);

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                const index = row * this.columns + col;
                if (col > row) {
                    result.values[index] = -Infinity;
                }
                else {
                    result.values[index] = this.values[index];
                }
            }
        }

        return result;
    }
}
