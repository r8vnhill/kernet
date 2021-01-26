/**
 * "kernet" (c) by Ignacio Slater M.
 * "kernet" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.kernet.functions

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TType

/**
 * Interface for all activation functions.
 *
 * Every class implementing this interface must provide a method to set the _features_ to compute
 * the function, and a method to execute (*call*) the computation.
 *
 * @param T
 *    the type of data used by the operation
 * @author Ignacio Slater Muñoz
 */
interface IActivationFunction<T : TType> {
  /**
   * Calculates the value of the activation function and returns the corresponding operand.
   *
   * @see [Operand]
   */
  fun call(): Operand<T>

  /**
   * Calculates the derivative of this activation function.
   */
  fun derivative(): Operand<T>

  /**
   * Calculates the value of the activation function for a given set of input features and returns
   * the corresponding operand.
   *
   * @see [Operand]
   */
  operator fun invoke(x: Operand<T>): Operand<T>

  /**
   * Set the inputs for the activation function.
   * This method returns the activation function to provide a fluid interface.
   */
  fun setFeatures(x: Operand<T>): IActivationFunction<T>
}

abstract class AbstractActivationFunction<T : TType>(protected val tf: Ops) :
  IActivationFunction<T> {
  lateinit var features: Operand<T>
    protected set
}