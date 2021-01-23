package cl.ravenhill.kernet.functions

import org.junit.jupiter.api.Assertions.assertTrue
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import kotlin.properties.Delegates
import kotlin.random.Random

/**
 * @author [Ignacio Slater Muñoz](mailto:ignacio.slater@ug.uchile.cl)
 */
internal class ActivationKtTest {
  private var seed by Delegates.notNull<Long>()
  private val tf = Ops.create()

  @org.junit.jupiter.api.BeforeEach
  fun setUp() {
    seed = Random.nextLong()
  }

  @org.junit.jupiter.api.Test
  fun `sigmoid function must be in range 0 to 1`() {
    val rng = Random(seed)
    for (i in 0 until 10) {
      val t = TFloat32.tensorOf(Shape.of(rng.nextLong(10), rng.nextLong(10), rng.nextLong(10)))
      t.data().scalars().forEach { scalar -> scalar.setFloat(rng.nextFloat() * 100) }
      val sig = sigmoid(t)
      sig.data().scalars().forEach { assertTrue(it.getFloat() in 0.0..1.0, "Test failed with seed: $seed. ${it.getFloat()} is not in [0, 1]") }
    }
  }
}