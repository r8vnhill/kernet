/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.functions

import cl.ravenhill.kernet.minus
import cl.ravenhill.kernet.tf
import org.tensorflow.ConcreteFunction
import org.tensorflow.Operand
import org.tensorflow.Signature
import org.tensorflow.Tensor
import org.tensorflow.ndarray.NdArrays
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

fun sigmoid(x: Tensor<TFloat32>): Tensor<TFloat32> {
  val signature = { ops: Ops ->
    tf = ops
    val placeholder = tf.placeholder(TFloat32.DTYPE)
    val a = tf.math.exp(placeholder).plus(1F)
//    val sig = tf.math.reciprocal(tf.math.exp(placeholder) + 1F)
    Signature.builder().input("x", placeholder).output("sigmoid", sig).build()
  }
  ConcreteFunction.create(signature).use { f ->
    return f.call(x).expect(TFloat32.DTYPE)
  }
}

fun main() {
  val t = TFloat32.tensorOf(NdArrays.ofFloats(Shape.scalar()))
  t.data().setFloat(0F)
  sigmoid(t).data().scalars().forEach { println(it.getFloat()) }
}
