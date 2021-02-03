/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
@file:Suppress("UNUSED_CHANGED_VALUE")

package cl.ravenhill.kernet.functions.activation

import cl.ravenhill.kernet.math.OperationsContext
import cl.ravenhill.kernet.math.max
import cl.ravenhill.kernet.math.times
import cl.ravenhill.kernet.math.exp
import cl.ravenhill.kernet.math.min
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mul
import org.tensorflow.types.TFloat32

/**
 * Wrapper class for the Swish activation function.
 *
 * @property tf
 *    the context for the operations
 * @see [Ops]
 */
class KSwish(tf: Ops, var beta: Float = 1F) : AbstractActivationFunction<TFloat32>(tf) {

  override fun call(): Mul<TFloat32> {
    OperationsContext.setOperatorContext(tf)
    return features * sigmoid(tf, beta * features)
  }

  override fun invoke(x: Operand<TFloat32>): Mul<TFloat32> {
    setFeatures(x)
    return call()
  }

  override fun setFeatures(x: Operand<TFloat32>): KSwish {
    features = x
    return this
  }
}

/**
 * Wrapper class for the CELU activation function.
 *
 * @property tf
 *    the context for the operations
 * @see [Ops]
 */
class KCELU(tf: Ops) : AbstractActivationFunction<TFloat32>(tf) {
  private var alpha = 1F

  override fun call(): Operand<TFloat32> {
    OperationsContext.setOperatorContext(tf)
    val zeros = tf.zerosLike(features)
    return max(zeros, features) + min(zeros, alpha * exp(features / alpha) - tf.onesLike(features))
  }

  override fun invoke(x: Operand<TFloat32>): Mul<TFloat32> {
    TODO()
//    return setFeatures(x).call()
  }

  fun invoke(x: Operand<TFloat32>, alpha: Float): Mul<TFloat32> {
    TODO()
//    return setFeatures(x).let {
//      this.alpha = alpha
//      this
//    }.call()
  }

  override fun setFeatures(x: Operand<TFloat32>): KCELU {
    features = x
    return this
  }
}

fun swish(tf: Ops, features: Operand<TFloat32>, beta: Float = 1F) =
  KSwish(tf, beta).invoke(features)

fun celu(tf: Ops, features: Operand<TFloat32>, alpha: Float = 1F) =
  KCELU(tf).invoke(features, alpha)

fun main() {
  val tf = Ops.create()
  swish(tf, tf.constant(1F))
}