/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.functions.activation

import cl.ravenhill.kernet.math.minus
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Sigmoid
import org.tensorflow.op.math.Tanh
import org.tensorflow.op.nn.Relu
import org.tensorflow.op.nn.Softmax
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType


/**
 * Wrapper class for the Sigmoid activation function.
 *
 * @param T
 *    the type of data used by the operation
 * @property tf
 *    the context for the operations
 * @see [Ops]
 * @see [Sigmoid]
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

  override fun derivative(): Operand<T> {
    TODO("Not yet implemented")
  }
}

/**
 * Wrapper class for the ReLU activation function.
 *
 * @param T
 *    the type of data used by the operation
 * @property tf
 *    the context for the operations
 * @see [Ops]
 * @see [Relu]
 */
class KReLU<T : TType>(tf: Ops) : AbstractActivationFunction<T>(tf) {
  override fun call(): Relu<T> = tf.nn.relu(features)

  override fun invoke(x: Operand<T>): Relu<T> {
    setFeatures(x)
    return call()
  }

  override fun setFeatures(x: Operand<T>): KReLU<T> {
    features = x
    return this
  }

  override fun derivative(): Operand<T> {
    TODO("Not yet implemented")
  }
}

/**
 * Wrapper class for the tanh activation function.
 *
 * @param T
 *    the type of data used by the operation
 * @property tf
 *    the context for the operations
 * @see [Ops]
 */
class KTanh<T : TType>(tf: Ops) : AbstractActivationFunction<T>(tf) {
  override fun call(): Tanh<T> = tf.math.tanh(features)

  override fun invoke(x: Operand<T>): Tanh<T> {
    setFeatures(x)
    return call()
  }

  override fun setFeatures(x: Operand<T>): KTanh<T> {
    features = x
    return this
  }

  override fun derivative(): Operand<T> = tf.onesLike(features) - tf.math.square(invoke(features))
}

/**
 * Wrapper class for the Softmax activation function.
 *
 * @param T
 *    the type of data used by the operation
 * @property tf
 *    the context for the operations
 * @see [Ops], [Softmax]
 */
class KSoftmax<T : TNumber>(tf: Ops) : AbstractActivationFunction<T>(tf) {
  override fun call(): Softmax<T> = tf.nn.softmax(features)

  override fun invoke(x: Operand<T>): Softmax<T> {
    setFeatures(x)
    return call()
  }

  override fun setFeatures(x: Operand<T>): KSoftmax<T> {
    features = x
    return this
  }

  override fun derivative(): Operand<T> {
    TODO("Not yet implemented")
  }
}

// region : activation functions
fun <T : TType> sigmoid(tf: Ops, features: Operand<T>) = KSigmoid<T>(tf).invoke(features)

fun <T : TType> relu(tf: Ops, features: Operand<T>) = KReLU<T>(tf).invoke(features)

fun <T : TType> tanh(tf: Ops, features: Operand<T>) = KTanh<T>(tf).invoke(features)

fun <T : TNumber> softmax(tf: Ops, features: Operand<T>) = KSoftmax<T>(tf).invoke(features)
// endregion