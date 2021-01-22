///**
// * "kernet" (c) by Ignacio Slater M.
// * "kernet" is licensed under a
// * Creative Commons Attribution 4.0 International License.
// * You should have received a copy of the license along with this
// * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
// */
//package cl.ravenhill.kernet.nn.layers
//
//import org.jetbrains.kotlinx.dl.api.core.KGraph
//import org.jetbrains.kotlinx.dl.api.core.layer.Layer
//import org.tensorflow.Operand
//import org.tensorflow.Shape
//import org.tensorflow.op.Ops
//
//class Concatenate(val axis: Int = -1, name: String = "") : Layer(name) {
//  val supportsMasking = true
//  private val reshapeRequired = false
//
//  override fun computeOutputShape(inputShape: Shape): Shape {
//    val outputShape = inputShape
//    TODO("Not yet implemented")
//  }
//
//  override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
//    TODO("Not yet implemented")
//  }
//
//  override fun getParams(): Int {
//    TODO("Not yet implemented")
//  }
//
//  override fun getWeights(): List<Array<*>> {
//    TODO("Not yet implemented")
//  }
//
//  override fun hasActivation(): Boolean {
//    TODO("Not yet implemented")
//  }
//
////  override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
////    TODO("Not yet implemented")
////  }
//
//}