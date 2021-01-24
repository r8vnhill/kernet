/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.math

import cl.ravenhill.kernet.math.OperatorContext.math
import cl.ravenhill.kernet.math.OperatorContext.tf
import org.tensorflow.Operand
import org.tensorflow.op.MathOps
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mul
import org.tensorflow.types.TFloat32
import org.tensorflow.types.family.TType

/**
 * Object re´resenting the environment where the operations are being executed.
 *
 * @property math
 *    the ``MathOps`` environment where the operations are executed.
 */
object OperatorContext {
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

/**
 * Multiplies two ``Operand`` objects and returns the result wrapped in a ``Mul`` operand.
 * @see [Mul]
 * @see [Operand]
 */
operator fun Operand<TFloat32>.times(x: Float): Mul<TFloat32> = math.mul(this, tf.constant(x))