/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.functions

import cl.ravenhill.kernet.math.OperatorContext
import cl.ravenhill.kernet.math.times
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Sigmoid
import org.tensorflow.op.math.Tanh
import org.tensorflow.op.nn.Relu
import org.tensorflow.op.nn.Softmax
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

/**
 * Interface for all activation functions.
 *
 * Every class implementing this interface must provide a method to set the _features_ to compute
 * the function, and a method to execute (*call*) the computation.
 *
 * @param T
 *    the type of data used by the operation
 * @author Ignacio Slater Muñoz
 */
interface IActivationFunction<T : TType> {
  /**
   * Calculates the value of the activation function and returns the corresponding operand.
   *
   * @see [Operand]
   */
  fun call(): Operand<T>

  /**
   * Calculates the value of the activation function for a given set of input features and returns
   * the corresponding operand.
   *
   * @see [Operand]
   */
  operator fun invoke(x: Operand<T>): Operand<T>

  /**
   * Set the inputs for the activation function.
   * This method returns the activation function to provide a fluid interface.
   */
  fun setFeatures(x: Operand<T>): IActivationFunction<T>
}

abstract class AbstractActivationFunction<T : TType>(protected val tf: Ops) :
  IActivationFunction<T> {
  lateinit var features: Operand<T>
    protected set
}

/**
 * Wrapper class for the Sigmoid activation function.
 * @param T
 *    the type of data used by the operation
 * @property tf
 *    the context for the operations
 * @see [Ops]
 */
class KSigmoid<T : TType>(tf: Ops) : AbstractActivationFunction<T>(tf) {
  override fun setFeatures(x: Operand<T>): KSigmoid<T> {
    features = x
    return this
  }

  override fun call(): Sigmoid<T> = tf.math.sigmoid(features)

  override operator fun invoke(x: Operand<T>): Sigmoid<T> {
    setFeatures(x)
    return call()
  }
}

fun <T : TType> sigmoid(tf: Ops, features: Operand<T>) = KSigmoid<T>(tf).invoke(features)

fun <T : TType> relu(tf: Ops, features: Operand<T>): Relu<T> = tf.nn.relu(features)

fun <T : TType> tanh(tf: Ops, features: Operand<T>): Tanh<T> = tf.math.tanh(features)

fun <T : TNumber> softmax(tf: Ops, features: Operand<T>): Softmax<T> = tf.nn.softmax(features)

fun <T : TType> swish(tf: Ops, features: Operand<T>, beta: Tensor<T>): Operand<T> {
  OperatorContext.setOperatorContext(tf)
  return tf.math.mul(features, sigmoid(tf, features * beta))
}


