/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.functions

import org.tensorflow.ConcreteFunction
import org.tensorflow.Operand
import org.tensorflow.Signature
import org.tensorflow.TensorFlow
import org.tensorflow.op.MathOps
import org.tensorflow.op.Ops
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TType

val math: MathOps = Ops.create().math

fun <T : TType> sigmoid(x: Operand<T>) = math.sigmoid(x)

fun main() {
  println(TensorFlow.version())
  val fn = ConcreteFunction.create(::dbl)
  val x = TInt32.scalarOf(10)

  fn.close()
}

fun dbl(tf: Ops): Signature {
  val x = tf.placeholder(TInt32.DTYPE)
  val dblX = tf.math.add(x, x)
  return Signature.builder().input("x", x).output("dbl", dblX).build()
}