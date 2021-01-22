/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.convolutional.models

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.MaxPool2D

/**
 * Inception module as proposed by Szegedy et.al. on its paper
 * [Going Deeper With Convolutions
](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)
 *
 * An inception module is a block of parallel convolutional layers with different sized filters and
 * a 3x3 max pooling layer, the results of which are then concatenated.
 *
 * @property branches
 *    an array with the parallel convolutional layers of the module.
 * @constructor
 */
private class InceptionModule(
  input: Input,
  filters1x1: Long,
  reduce3x3: Long,
  filters3x3: Long,
  reduce5x5: Long,
  filter5x5: Long,
  filtersPoolProj: Long
) {
  private val branches = arrayOf(
    Sequential.of(
      input,
      Conv2D(filters = filters1x1, kernelSize = longArrayOf(1, 1))
    ),
    Sequential.of(
      input,
      Conv2D(reduce3x3, kernelSize = longArrayOf(1, 1)),
      Conv2D(filters3x3, kernelSize = longArrayOf(3, 3))
    ),
    Sequential.of(
      input,
      Conv2D(reduce5x5, kernelSize = longArrayOf(1, 1)),
      Conv2D(filter5x5, kernelSize = longArrayOf(3, 3))
    ),
    Sequential.of(
      input,
      MaxPool2D(poolSize = intArrayOf(3, 3), strides = intArrayOf(1, 1)),
      Conv2D(filtersPoolProj, kernelSize = longArrayOf(1, 1))
    )
  )

  fun forward(x: Input): Unit {

    return
  }
}