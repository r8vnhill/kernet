import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File

/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */

val floatArray = ImageConverter.toRawFloatArray(File("datasets/mnist/test-image-bag.png"))

fun reshapeInput(inputData: FloatArray): Array<Array<FloatArray>> {
  val reshaped = Array(1) { Array(28) { FloatArray(28) } }
  inputData.indices.forEach { reshaped[0][it / 28][it % 28] = inputData[it] }
  return reshaped
}

fun main() {
  val model = InferenceModel.load(File("src/model/my_model"))
  model.reshape(::reshapeInput)
  val prediction = model.predict(floatArray)
  println("Predicted label is: $prediction.")
  model.close()
}