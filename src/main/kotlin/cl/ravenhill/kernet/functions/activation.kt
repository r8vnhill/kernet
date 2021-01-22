/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.functions

import cl.ravenhill.kernet.tf
import org.tensorflow.EagerSession
import org.tensorflow.Signature
import org.tensorflow.Tensor
import org.tensorflow.ndarray.IntNdArray
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32

fun sigmoid(x: Tensor<TFloat32>): Tensor<TInt32> {
  EagerSession.create().use {
    val tf = Ops.create(it)

  }
  val one = TInt32.scalarOf(1)
  return one
}

private fun sigmoidSignature(ops: Ops): Signature {
  tf = ops
  val x = tf.placeholder(TFloat32.DTYPE)
  val sig = x + 1F
  return Signature.builder().input("x", x).output("sigmoid", sig).build()
}

fun main() {
  // Allocate a tensor of 32-bits integer of the shape (2, 3, 2)
  val tensor: Tensor<TInt32> = TInt32.tensorOf(Shape.of(2, 3, 2))

// Access tensor memory directly
  val tensorData: IntNdArray = tensor.data()

  EagerSession.create().use { session ->
    val tf = Ops.create(session)
    val one = tf.constant(1)
    one.data().scalars().forEach { println(it.getInt()) }
  }
}
