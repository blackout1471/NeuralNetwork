using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Maf
{
    public class Matrix
    {

        public delegate float MapFunc(float value);

        #region Getters / Setters

        /// <summary>
        /// Get a new Matrix which is transposed
        /// </summary>
        public Matrix Transposed
        {
            get
            {
                Matrix _res = Duplicate();
                _res.Transpose();

                return _res;
            }
        }

        /// <summary>
        /// Get the average values for all the values in the matrix
        /// </summary>
        public float AverageValue
        {
            get
            {
                float v = 0f;

                for (int i = 0; i < _data.Length; i++)
                {
                    v += _data[i];
                }

                v /= _data.Length;

                return v;
            }
        }

        public float Squeeze
        {
            get
            {
                float v = 0f;

                for (int i = 0; i < _data.Length; i++)
                {
                    v += _data[i];
                }

                return v;
            }
        }

        /// <summary>
        /// The rows in the matrix
        /// </summary>
        public int Rows { get => rows; }
        private int rows;

        /// <summary>
        /// The columns in the matrix
        /// </summary>
        public int Columns { get => cols; }
        private int cols;


        /// <summary>
        /// Get the matrix data or set it
        /// </summary>
        /// <param name="_row"></param>
        /// <param name="_col"></param>
        /// <returns></returns>
        public float this[int _row, int _col]
        {
            get
            {
                return _data[_row * Columns + _col];
            }
            set
            {
                _data[_row * Columns + _col] = value;
            }
        }
        private float[] _data;

        #endregion

        #region Constructors

        /// <summary>
        /// Create a matrix with exact rows and cols
        /// all values are 0
        /// </summary>
        /// <param name="_rows"></param>
        /// <param name="_cols"></param>
        public Matrix(int _rows, int _cols)
        {
            rows = _rows;
            cols = _cols;

            _data = new float[rows * cols];
        }

        /// <summary>
        /// Create a matrix based on values
        /// </summary>
        /// <param name="_rows"></param>
        /// <param name="_cols"></param>
        /// <param name="_values"></param>
        public Matrix(int _rows, int _cols, float[] _values) : this(_rows, _cols)
        {
            if (rows * cols > _values.Length)
                throw new InvalidOperationException("There is too many inputs for the current matrix");

            _data = _values;
        }

        /// <summary>
        /// Create a matrix based on a 2d array
        /// </summary>
        /// <param name="_values"></param>
        public Matrix(float[,] _values) : this(_values.GetLength(0), _values.GetLength(1))
        {
            float[] _rm = new float[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int k = 0; k < cols; k++)
                {
                    _rm[i * cols + k] = _values[i, k];
                }
            }

            _data = _rm;
        }

        #endregion

        #region Local operator methods
        /// <summary>
        /// Add a value to all the matrix data
        /// </summary>
        /// <param name="_val"></param>
        public void Add(float _val)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] += _val;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="_m"></param>
        public void Add(Matrix _m)
        {
            if (Rows != _m.Rows || Columns != _m.Columns)
                throw new Exception("The 2 matrices is not of the same sizes");

            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] += _m._data[i];
            }
        }

        /// <summary>
        /// Subtract a value from all the matrix data
        /// </summary>
        /// <param name="_val"></param>
        public void Sub(float _val)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] -= _val;
            }
        }

        /// <summary>
        /// Subtract a matrix from another
        /// </summary>
        /// <param name="_m"></param>
        public void Sub(Matrix _m)
        {
            if (Rows != _m.Rows || Columns != _m.Columns)
                throw new Exception("The 2 matrices is not of the same sizes");

            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] -= _m._data[i];
            }
        }

        /// <summary>
        /// Divide all the values in the matrix with the value
        /// </summary>
        /// <param name="_val"></param>
        public void Div(float _val)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                if (_data[i] == 0)
                    continue;

                _data[i] /= _val;
            }
        }

        /// <summary>
        /// Divide a matrix with another matrix
        /// </summary>
        /// <param name="_m"></param>
        public void Div(Matrix _m)
        {
            if (Rows != _m.Rows || Columns != _m.Columns)
                throw new Exception("The 2 matrices is not of the same sizes");

            for (int i = 0; i < _data.Length; i++)
            {
                if (_data[i] == 0 || _m._data[i] == 0)
                    continue;

                _data[i] /= _m._data[i];
            }
        }

        /// <summary>
        /// Multiply a value to all the matrix data
        /// </summary>
        /// <param name="_val"></param>
        public void Multiply(float _val)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] *= _val;
            }
        }

        /// <summary>
        /// The matrix multiplication
        /// </summary>
        /// <param name="_m"></param>
        public void Multiply(Matrix _m)
        {
            if (Columns != _m.Rows)
                throw new Exception("The Columns of matrix a and the rows of matrix b is not equal");

            float[] res = new float[Rows * _m.Columns];

            for (int i = 0; i < Rows; i++)
            {
                for (int k = 0; k < _m.Columns; k++)
                {
                    float sum = 0f;

                    for (int j = 0; j < _m.Rows; j++)
                    {
                        sum += this[i, j] * _m[j, k];
                    }

                    res[i * _m.Columns + k] = sum;
                }
            }

            rows = Rows;
            cols = _m.Columns;
            _data = res;
        }

        /// <summary>
        /// The hadamar product between the current matrix and the other
        /// </summary>
        /// <param name="_m"></param>
        public void Hadamar(Matrix _m)
        {
            if (Rows != _m.Rows || Columns != _m.Columns)
                throw new Exception("The 2 matrices is not of the same sizes");

            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] *= _m._data[i];
            }
        }

        /// <summary>
        /// Map a function to the matrix
        /// </summary>
        /// <param name="_func"></param>
        public void Map(MapFunc _func)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = _func(_data[i]);
            }
        }

        #endregion

        #region Logic operations
        /// <summary>
        /// Transpose the current matrix
        /// </summary>
        public void Transpose()
        {
            float[] res = new float[rows * cols];

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    res[i * rows + j] = this[j, i];
                }
            }

            int _col = rows;
            int _row = cols;

            rows = _row;
            cols = _col;
            _data = res;
        }

        /// <summary>
        /// Duplicate a matrix
        /// </summary>
        /// <returns></returns>
        public Matrix Duplicate()
        {
            Matrix _res = new Matrix(Rows, Columns);

            for (int i = 0; i < _res._data.Length; i++)
            {
                _res._data[i] = _data[i];
            }

            return _res;
        }

        /// <summary>
        /// Randomize a new matrix between 0.0f -> 1.0f
        /// </summary>
        public void Randomize()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = (float)rnd.NextDouble();
            }
        }

        /// <summary>
        /// Randomize between numbers
        /// </summary>
        /// <param name="_start"></param>
        /// <param name="_end"></param>
        public void Randomize(int _start, int _end)
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = (float)rnd.Next(_start, _end);
            }
        }

        /// <summary>
        /// Randomize between numbers then divide the number
        /// </summary>
        /// <param name="_start"></param>
        /// <param name="_end"></param>
        /// <param name="divider"></param>
        public void Randomize(int _start, int _end, int _divider)
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = (float)rnd.Next(_start, _end) / _divider;
            }
        }

        #endregion

        #region static operators

        /// <summary>
        /// Add matrixes together returns a new
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix _a, Matrix _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Add(_b);

            return _res;
        }

        /// <summary>
        /// Add a matrix and a value together returns a new matrix
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix _a, float _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Add(_b);

            return _res;
        }

        /// <summary>
        /// Add a matrix and a value together returns a new matrix
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator +(float _b, Matrix _a)
        {
            Matrix _res = _a.Duplicate();
            _res.Add(_b);

            return _res;
        }

        /// <summary>
        /// Subtract matrixes together returns a new
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator -(Matrix _a, Matrix _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Sub(_b);

            return _res;
        }


        /// <summary>
        /// Subtract a Matrix and a value from the matrix
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator -(Matrix _a, float _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Sub(_b);

            return _res;
        }

        /// <summary>
        /// Divide matrixes together returns a new
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator /(Matrix _a, Matrix _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Div(_b);

            return _res;
        }

        /// <summary>
        /// Divide a matrix and a value
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator /(Matrix _a, float _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Div(_b);

            return _res;
        }

        /// <summary>
        /// Multiply matrixes together returns a new
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator *(Matrix _a, Matrix _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Multiply(_b);

            return _res;
        }

        /// <summary>
        /// Multiply a scalar to a matrix returns a new matrix
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator *(Matrix _a, float _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Multiply(_b);

            return _res;
        }

        /// <summary>
        /// Multiply a scalar to a matrix returns a new matrix
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix operator *(float _b, Matrix _a)
        {
            Matrix _res = _a.Duplicate();
            _res.Multiply(_b);

            return _res;
        }

        /// <summary>
        /// Hadamar 2 matrixes
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_b"></param>
        /// <returns></returns>
        public static Matrix Hadamar(Matrix _a, Matrix _b)
        {
            Matrix _res = _a.Duplicate();
            _res.Hadamar(_b);

            return _res;
        }

        /// <summary>
        /// Map a matrix to a function a get a new matrix
        /// </summary>
        /// <param name="_a"></param>
        /// <param name="_func"></param>
        /// <returns></returns>
        public static Matrix Map(Matrix _a, MapFunc _func)
        {
            Matrix _res = _a.Duplicate();
            _res.Map(_func);

            return _res;
        }
        #endregion



        /// <summary>
        /// Get the matrix as a string
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string s = "";

            for (int i = 0; i < Rows; i++)
            {
                for (int k = 0; k < Columns; k++)
                {
                    s += this[i, k] + " ";
                }

                s += "\n";
            }

            return s;
        }

    }
}
