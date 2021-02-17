/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.math

import cl.ravenhill.kernet.math.OperationsContext.math
import cl.ravenhill.kernet.math.OperationsContext.tf
import org.tensorflow.Operand
import org.tensorflow.op.MathOps
import org.tensorflow.op.Ops
import org.tensorflow.op.math.*
import org.tensorflow.types.TFloat32
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

/**
 * Object re´resenting the environment where the operations are being executed.
 *
 * @property math
 *    the ``MathOps`` environment where the operations are executed.
 */
object OperationsContext {
  lateinit var tf: Ops
    private set
  lateinit var math: MathOps
    private set

  /**
   * Sets the context of the operations.
   *
   * @see [Ops]
   * @see [MathOps]
   */
  fun setOperatorContext(tf: Ops) {
    this.tf = tf
    math = tf.math
  }
}

operator fun <T : TType> Operand<T>.plus(x: Operand<T>): Add<T> = math.add(this, x)

operator fun <T : TType> Operand<T>.minus(x: Operand<T>): Sub<T> = math.sub(this, x)

/**
 * Multiplies a float by a  and returns the result wrapped in a ``Mul`` operand.
 * @see [Mul]
 * @see [Operand]
 */
operator fun Float.times(x: Operand<TFloat32>): Mul<TFloat32> = math.mul(tf.constant(this), x)

operator fun <T : TType> Operand<T>.times(x: Operand<T>): Mul<T> = math.mul(this, x)

/**
 * Element-wise maximum between two numerical operands.
 *
 * @see [Maximum]
 * @see [Operand]
 * @see [TNumber]
 */
fun <T : TNumber> max(a: Operand<T>, b: Operand<T>): Maximum<T> = math.maximum(a, b)

/**
 * Element-wise minimum between two numerical operands.
 *
 * @see [Minimum]
 * @see [Operand]
 * @see [TNumber]
 */
fun <T : TNumber> min(a: Operand<T>, b: Operand<T>): Minimum<T> = math.minimum(a, b)

/**
 * Returns the result of applying the exponential function element-wise to an operand.
 *
 * @see [Exp]
 * @see [TNumber]
 * @see [Operand]
 */
fun <T: TNumber> exp(x: Operand<T>): Exp<T> = math.exp(x)