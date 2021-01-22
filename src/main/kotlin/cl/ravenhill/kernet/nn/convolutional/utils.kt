/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.nn.convolutional

/**
 * Utility class representing a 2D padding with arbitrary heights and widths.
 */
class Padding2D(topPad: Int, bottomPad: Int, leftPad: Int, rightPad: Int) {
  private val _pad = arrayOf(topPad, bottomPad, leftPad, rightPad)
  val pad = _pad.copyOf()
  val heights = Pair(_pad[0], _pad[1])
  val widths = Pair(_pad[2], _pad[3])
}
