using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNet.Maf;

namespace NeuralNet
{
    class Program
    {
        private static int iteration = 0;

        static void Main(string[] args)
        {
            //float[,] _inp = {
            //    { 0, 0 },
            //    { 0, 1 },
            //    { 1, 1 },
            //    { 1, 0 },
            //};

            //float[,] _ans =
            //{
            //    { 0 },
            //    { 1 },
            //    { 0 },
            //    { 1 }
            //};

            Console.WriteLine("Loading Network data...");

            string[] _sInp = File.ReadAllLines(@"C:\Users\Emil\Downloads\mnist-in-csv\mnist_train.csv");

            string[][] s = _sInp.Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();

            var b = s.Select(x => new
            {
                Answer = int.Parse(x[0]),
                Inputs = x.Skip(1).ToArray()
            }).ToArray();

            float[,] _inp = new float[b.GetLength(0), b[0].Inputs.Length];
            float[,] _ans = new float[b.GetLength(0), 10];

            for (int i = 0; i < _inp.GetLength(0); i++)
            {
                _ans[i, b[i].Answer] = 0.99f;

                for (int k = 0; k < _inp.GetLength(1); k++)
                {
                    _inp[i, k] = float.Parse(b[i].Inputs[k]) / 255;
                }
            }

            NeuralNet a = new NeuralNet(784, 100, 10)
            {
                BatchSize = 200,
                LearningRate = 0.3f,
                ShuffleData = true,
            };

            a.SetMonitor(DisplayNetwork);
            a.Training(_inp, _ans, 18);

            Console.WriteLine("Guessing...");

            Console.WriteLine(a.Guess(a.SliceArr(_inp, 5)));
            Console.WriteLine(new Matrix(1, 10, a.SliceArr(_ans, 5)));
            Console.WriteLine(a.Guess(a.SliceArr(_inp, 1)));
            Console.WriteLine(new Matrix(1, 10, a.SliceArr(_ans, 1)));
            Console.WriteLine(a.Guess(a.SliceArr(_inp, 2)));
            Console.WriteLine(new Matrix(1, 10, a.SliceArr(_ans, 2)));

            Console.ReadKey();
        }

        static void DisplayNetwork(Matrix _i, Matrix _o, Matrix _er, Matrix _ex)
        {
            iteration++;
            int top = 0;

            if (iteration % 10 != 0)
                return;

            //Console.Clear();

            //Console.WriteLine("Iteration: " + iteration + "\n_______________");
            //Console.WriteLine("Output\n___________\n" + _o);
            //Console.WriteLine("Expected\n__________\n" + _ex);
            //Console.WriteLine("Error\n_________\n" + (Math.Abs(_er.Squeeze * 100)) + "%");

            // Iteration
            Console.SetCursorPosition(0, top);
            Console.Write("Iteration: " + iteration + "\n_________________");
            top += 2;

            //// Input writeline
            //Console.SetCursorPosition(0, top);
            //Console.Write("Input");
            //top += 2;

            //// Input
            //Console.SetCursorPosition(0, top);
            //Console.Write(_i);
            //top += _i.Rows;

            Console.SetCursorPosition(0, top);
            Console.Write("_________________");
            top++;

            // output writeline
            Console.SetCursorPosition(0, top);
            Console.Write("output");
            top += 2;

            // Output
            Console.SetCursorPosition(0, top);
            Console.Write(_o);
            top += _o.Rows;

            Console.SetCursorPosition(0, top);
            Console.Write("_________________");
            top++;

            // expected writeline
            Console.SetCursorPosition(0, top);
            Console.Write("Expected");
            top += 2;

            // Expected
            Console.SetCursorPosition(0, top);
            Console.Write(_ex);
            top += _ex.Rows;

            Console.SetCursorPosition(0, top);
            Console.Write("_________________");
            top++;

            // Input writeline
            Console.SetCursorPosition(0, top);
            Console.Write("Error");
            top += 2;

            // Error
            Console.SetCursorPosition(0, top);
            Console.Write(Math.Abs(_er.Squeeze));
            top += _er.Rows;


        }
    }
}
