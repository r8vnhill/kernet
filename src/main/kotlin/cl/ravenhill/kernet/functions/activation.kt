/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.functions

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Tanh
import org.tensorflow.op.nn.Relu
import org.tensorflow.op.nn.Softmax
import org.tensorflow.types.family.TNumber
import org.tensorflow.types.family.TType

fun <T : TType> sigmoid(tf: Ops, features: Operand<T>): Operand<T> {
  return tf.math.sigmoid(features)
}

fun <T : TType> relu(tf: Ops, features: Operand<T>): Relu<T> = tf.nn.relu(features)

fun <T : TType> tanh(tf: Ops, features: Operand<T>): Tanh<T> = tf.math.tanh(features)

fun <T : TNumber> softmax(tf: Ops, features: Operand<T>): Softmax<T> = tf.nn.softmax(features)

fun <T : TType> swish(tf: Ops, features: Operand<T>, beta: Tensor<T>): Operand<T> =
  tf.math.mul(features, tf.math.sigmoid(tf.math.mul(features, tf.constant(beta))))


