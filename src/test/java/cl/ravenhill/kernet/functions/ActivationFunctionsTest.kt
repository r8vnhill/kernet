package cl.ravenhill.kernet.functions

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.RepeatedTest
import org.tensorflow.ndarray.FloatNdArray
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.types.TFloat32
import kotlin.math.abs
import kotlin.math.exp
import kotlin.random.Random

/**
 * @author [Ignacio Slater Muñoz](mailto:ignacio.slater@ug.uchile.cl)
 */
internal class ActivationFunctionsTest {
  private lateinit var tf: Ops
  private var seed = 0
  private val eps = 1e-3

  @BeforeEach
  fun setUp() {
    tf = Ops.create()
    seed = Random.nextInt()
  }

  @RepeatedTest(16)
  fun `sigmoid function result is in range 0 to 1`() {
    checkActivationFunction { _, it ->
      assertTrue(
        it.getFloat() in 0.0..1.0,
        "Test failed with seed: $seed. ${it.getFloat()} is not in [0, 1]"
      )
    }
  }

  @RepeatedTest(16)
  fun `sigmoid results matches function definition`() {
    checkActivationFunction { x, it ->
      val expected = 1 / (1 + exp(-x))
      assertTrue(
        abs(expected - it.getFloat()) < eps,
        "Test failed with seed: $seed. Expected: $expected but got ${it.getFloat()}"
      )
    }
  }

  private fun checkActivationFunction(assertFor: (Float, FloatNdArray) -> Unit) {
    val rng = Random(seed)
    val t = randomTensor(rng)
    val sig = sigmoid(tf, t).asTensor()
    sig.data().scalars().forEachIndexed { index, it -> assertFor(t.data().getFloat(*index), it) }
  }

  private fun randomTensor(rng: Random): Constant<TFloat32> {
    val shape = LongArray(rng.nextInt(4) + 1) {
      rng.nextLong(1, 10)
    }
    val t = TFloat32.tensorOf(Shape.of(*shape))
    t.data().scalars().forEach { scalar -> scalar.setFloat(rng.nextFloat() * 100) }
    return tf.constant(t)
  }
}