using NeuralNet.Maf;
using System;

namespace NeuralNet
{
    public class NeuralNet
    {
        public delegate void Monitor(Matrix _input, Matrix _output, Matrix _error, Matrix expected);

        private int inputCount;
        private int hiddenCount;
        private int outputCount;

        private Matrix weight_ih;
        private Matrix weight_oh;

        private Matrix bias_h;
        private Matrix bias_o;

        private Monitor monitor = null;

        public float LearningRate { get; set; } = 0.3f;
        public int BatchSize { get; set; } = 200;
        public bool ShuffleData { get; set; } = true;

        public NeuralNet(int _inputCount, int _hiddenCount, int _outputCount)
        {
            inputCount = _inputCount;
            hiddenCount = _hiddenCount;
            outputCount = _outputCount;

            weight_ih = new Matrix(hiddenCount, inputCount);
            weight_oh = new Matrix(outputCount, hiddenCount);
            weight_ih.Randomize(-50, 50, 100);
            weight_oh.Randomize(-50, 50, 100);

            bias_h = new Matrix(hiddenCount, 1);
            bias_o = new Matrix(outputCount, 1);
            bias_h.Randomize();
            bias_o.Randomize();
        }

        /// <summary>
        /// Set the monitoring method for the training
        /// </summary>
        /// <param name="_mon"></param>
        public void SetMonitor(Monitor _mon)
        {
            monitor += _mon;
        }

        /// <summary>
        /// Take a guess with the network
        /// </summary>
        /// <param name="_input"></param>
        /// <returns>the outputs</returns>
        public Matrix Guess(float[] _input)
        {
            Matrix inputs = new Matrix(inputCount, 1, _input);

            Matrix hidden = weight_ih * inputs;
            hidden.Add(bias_h);
            hidden.Map(MathHelper.Sigmoid);

            Matrix outputs = weight_oh * hidden;
            outputs.Add(bias_o);
            outputs.Map(MathHelper.Sigmoid);

            return outputs;
        }

        /// <summary>
        /// Train the network once
        /// </summary>
        /// <param name="_input"></param>
        /// <param name="_expectedOutput"></param>
        /// <param name="learningRate"></param>
        /// <returns></returns>
        public Matrix Train(float[] _input, float[] _expectedOutput)
        {
            // Convert float arrays
            Matrix inputs = new Matrix(inputCount, 1, _input);
            Matrix expected = new Matrix(outputCount, 1, _expectedOutput);

            // Feed forward
            Matrix hidden = weight_ih * inputs;
            hidden.Add(bias_h);
            hidden.Map(MathHelper.Sigmoid);

            Matrix outputs = weight_oh * hidden;
            outputs.Add(bias_o);
            outputs.Map(MathHelper.Sigmoid);

            //// Backpropagation

            //// Get error rate
            Matrix out_error = expected - outputs;
            Matrix hid_error = weight_oh.Transposed * out_error;

            // Calculate gradient

            Matrix gradient_oh = Matrix.Hadamar((LearningRate * out_error), Matrix.Map(outputs, MathHelper.PrimedSigmoid));
            Matrix gradient_ih = Matrix.Hadamar((LearningRate * hid_error), Matrix.Map(hidden, MathHelper.PrimedSigmoid));

            // adjust Bias
            bias_o.Add(gradient_oh);
            bias_h.Add(gradient_ih);

            // Adjust bias
            weight_oh.Add(gradient_oh * hidden.Transposed);
            weight_ih.Add(gradient_ih * inputs.Transposed);

            monitor?.Invoke(inputs, outputs, out_error, expected);

            return out_error;
        }

        /// <summary>
        /// Lets the network be trained in a sequence
        /// </summary>
        /// <param name="_inputs"></param>
        /// <param name="_answers"></param>
        /// <param name="_learningRate"></param>
        /// <param name="_epochs"></param>
        /// <param name="shuffleData"></param>
        public void Training(float[,] _inputs, float[,] _answers, int _epochs)
        {
            int _batches = (ShuffleData) ? BatchSize : _inputs.GetLength(0);

            for (int i = 0; i < _epochs; i++)
            {
                for (int k = 0; k < _batches; k++)
                {
                    int index = k;

                    if (ShuffleData)
                        index = new Random(Guid.NewGuid().GetHashCode()).Next(0, _inputs.GetLength(0));

                    float[] curInput = SliceArr(_inputs, index);
                    float[] curAnswer = SliceArr(_answers, index);

                    Train(curInput, curAnswer);
                }
            }
        }

        /// <summary>
        /// Method for splitting a 2d array
        /// </summary>
        /// <param name="input"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public float[] SliceArr(float[,] input, int index)
        {
            float[] _res = new float[input.GetLength(1)];

            for (int i = 0; i < input.GetLength(1); i++)
            {
                _res[i] = input[index, i];
            }

            return _res;
        }
    }
}
