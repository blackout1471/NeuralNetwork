using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Maf
{
    public static class MathHelper
    {
        /// <summary>
        /// returns a value that has been sigmoided
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static float Sigmoid(float val)
        {
            return 1f / (1f + ((float)Math.Pow(Math.E, -val)));
        }

        /// <summary>
        /// Returns the value which has been in the sigmoid prime method
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static float SigmoidPrime(float val)
        {
            return Sigmoid(val) * (1 - Sigmoid(val));
        }

        /// <summary>
        /// Returns the value of a sigmoid value to a sigmoid prime value
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static float PrimedSigmoid(float val)
        {
            return val * (1 - val);
        }

    }
}
