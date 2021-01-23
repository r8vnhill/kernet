/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import org.tensorflow.types.family.TType

lateinit var tf: Ops

operator fun Operand<TFloat32>.plus(scalar: Float) = this + tf.constant(scalar)


operator fun Operand<TFloat32>.plus(scalar: Int) = this + scalar.toFloat()

operator fun <T : TType?> Operand<T>.plus(op: Operand<T>): Operand<T> = tf.math.add(this, op)

operator fun Operand<TFloat32>.minus(scalar: Int) = this.plus(-scalar.toFloat())