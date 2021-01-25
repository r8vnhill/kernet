package cl.ravenhill.kernet.functions

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.RepeatedTest
import org.tensorflow.Operand
import org.tensorflow.ndarray.FloatNdArray
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.types.TFloat32
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random

/**
 * @author [Ignacio Slater Muñoz](mailto:ignacio.slater@ug.uchile.cl)
 */
internal class ActivationFunctionsTest {
  private val eps = 1e-3
  private var seed = 0
  private lateinit var rng: Random
  private lateinit var tf: Ops

  @BeforeEach
  fun setUp() {
    tf = Ops.create()
    seed = Random.nextInt()
    rng = Random(seed)
  }

  // region : invariants
  @RepeatedTest(16)
  fun `sigmoid function result is in range 0 to 1`() {
    val sigmoid = KSigmoid<TFloat32>(tf)
    checkActivationFunction(sigmoid) { _, it ->
      assertTrue(
        it.getFloat() in 0.0..1.0,
        "Test failed with seed: $seed. ${it.getFloat()} is not in [0, 1]"
      )
    }
  }

  @RepeatedTest(16)
  fun `ReLU should be greater or equal to 0`() {
    val relu = KReLU<TFloat32>(tf)
    checkActivationFunction(relu) { _, it ->
      assertTrue(
        it.getFloat() >= 0,
        "Test failed with seed: $seed. ${it.getFloat()} is negative"
      )
    }
  }

  @RepeatedTest(16)
  fun `tanh function result is in range -1 to 1`() {
    val tanh = KTanh<TFloat32>(tf)
    checkActivationFunction(tanh) { _, it ->
      assertTrue(
        it.getFloat() in -1.0..1.0,
        "Test failed with seed: $seed. ${it.getFloat()} is not in [-1, 1]"
      )
    }
  }

  @RepeatedTest(16)
  fun `softmax function result is in range 0 to 1`() {
    val softmax = KSoftmax<TFloat32>(tf)
    checkActivationFunction(softmax) { _, it ->
      assertTrue(
        it.getFloat() in 0.0..1.0,
        "Test failed with seed: $seed. ${it.getFloat()} is not in [0, 1]"
      )
    }
  }
  // endregion

  // region : computations
  @RepeatedTest(16)
  fun `sigmoid results matches function definition`() {
    val sigmoid = KSigmoid<TFloat32>(tf)
    checkActivationFunction(sigmoid) { x, it ->
      val expected = 1 / (1 + exp(-x))
      assertTrue(
        abs(expected - it.getFloat()) < eps,
        "Test failed with seed: $seed. Expected: $expected but got ${it.getFloat()}"
      )
    }
  }

  @RepeatedTest(16)
  fun `ReLU result matches function definition`() {
    val relu = KReLU<TFloat32>(tf)
    checkActivationFunction(relu) { x, it ->
      val expected = max(0F, x)
      assertTrue(
        abs(expected - it.getFloat()) < eps,
        "Test failed with seed: $seed. Expected: $expected but got ${it.getFloat()}"
      )
    }
  }
//
  @RepeatedTest(16)
  fun `tanh result matches function definition`() {
  val tanh = KTanh<TFloat32>(tf)
  checkActivationFunction(tanh) { x, it ->
      val expected = kotlin.math.tanh(x)
      assertTrue(
        abs(expected - it.getFloat()) < eps,
        "Test failed with seed: $seed. Expected: $expected but got ${it.getFloat()}"
      )
    }
  }

  @RepeatedTest(16)
  fun `swish result matches function definition`() {
    val beta = rng.nextFloat() * 100 - 50
    val swish = KSwish(tf, beta)
    checkActivationFunction(swish) { x, it ->
      val expected = x / (1 + exp(beta * x))
      assertTrue(
        abs(expected - it.getFloat()) < eps,
        "Test failed with seed: $seed. Expected: $expected but got ${it.getFloat()}"
      )
    }
  }
  // endregion

  private fun checkActivationFunction(
    function: IActivationFunction<TFloat32>,
    assertFor: (Float, FloatNdArray) -> Unit
  ) {
    val t = randomTensor(rng)
    val result = function(t)
    // FIXME: Esta wea no funciona, `it` entrega pura mierda
    result.data().scalars().forEachIndexed { index, it -> assertFor(t.data().getFloat(*index), it) }
  }

  private fun randomTensor(rng: Random): Constant<TFloat32> {
    val shape = LongArray(rng.nextInt(4) + 1) {
      rng.nextLong(1, 10)
    }
    val t = TFloat32.tensorOf(Shape.of(*shape))
    t.data().scalars().forEach { scalar -> scalar.setFloat(rng.nextFloat() * 100 - 50) }
    return tf.constant(t)
  }
}