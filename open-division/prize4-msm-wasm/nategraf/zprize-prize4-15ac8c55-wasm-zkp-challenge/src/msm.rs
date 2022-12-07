pub use ark_bls12_381::{G1Affine, G1Projective};
use ark_ec::{msm, AffineCurve, ProjectiveCurve};
use ark_ff::{fields::BitIteratorLE, PrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
#[cfg(feature = "std")]
use blake3::Hash;
#[cfg(feature = "std")]
use bytes::BufMut;
#[cfg(feature = "std")]
use std::fs::{create_dir_all, File};
#[cfg(feature = "std")]
use std::path::Path;

#[cfg(feature = "std")]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not serialize")]
    SerializationError(#[from] ark_serialize::SerializationError),

    #[error("io error")]
    IoError(#[from] std::io::Error),
}

// Define ScalarField and BigInt type aliases to avoid lengthy fully-qualified names.
pub type ScalarField = <G1Affine as AffineCurve>::ScalarField;
pub type BigInt = <ScalarField as PrimeField>::BigInt;

/// A struct wrapping the input for an msm problem
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Instance {
    pub points: Vec<G1Affine>,
    pub scalars: Vec<BigInt>,
}

impl Instance {
    pub fn generate(size: usize) -> Self {
        let (points, scalars) = generate_msm_inputs(size);
        Self { points, scalars }
    }

    pub fn compute_msm_baseline(&self) -> G1Projective {
        compute_msm_baseline(&self.points, &self.scalars)
    }

    pub fn compute_msm<const COMPLETE: bool, const BATCH_ACC_BUCKETS: bool>(&self) -> G1Projective {
        compute_msm::<COMPLETE, BATCH_ACC_BUCKETS>(&self.points, &self.scalars, None)
    }

    /// Get the size of the instance
    pub fn size(&self) -> usize {
        self.points.len()
    }
}

pub fn generate_msm_inputs(size: usize) -> (Vec<G1Affine>, Vec<BigInt>) {
    let mut rng = ark_std::test_rng();

    let scalar_vec = (0..size)
        .map(|_| ScalarField::rand(&mut rng).into_bigint())
        .collect::<Vec<_>>();

    // Vector of multiples 2^i & G_1, used to precompute the "doubling" portion of double and add.
    // TODO(victor): This could be improved by implementing a more optimal fixed base multiplcation
    // routine such as fixed base comb.
    let g_multiples = {
        let mut x = G1Projective::prime_subgroup_generator();
        let mut multiples = vec![x];

        // TODO: Don't hardcode that constant.
        for _ in 0..ScalarField::MODULUS_BIT_SIZE {
            x.double_in_place();
            multiples.push(x);
        }
        G1Projective::batch_normalization_into_affine(&multiples)
    };

    // Generate a number of random multipliers to apply to G_1 to generate a set of random bases.
    let factor_vec = (0..size)
        .map(|_| ScalarField::rand(&mut rng).into_bigint())
        .collect::<Vec<_>>();

    // Compute the multiples of G_1 using the precomputed tables of 2^i multiples.
    let point_vec = factor_vec
        .iter()
        .map(|r| {
            let bits = BitIteratorLE::new(r);
            let mut p = G1Projective::zero();
            for (i, b) in bits.enumerate() {
                if b {
                    p.add_assign_mixed(&g_multiples[i]);
                }
            }
            p
        })
        .collect::<Vec<_>>();

    let point_vec = G1Projective::batch_normalization_into_affine(&point_vec);
    return (point_vec, scalar_vec);
}

/// Currently using Pippenger's algorithm for multi-scalar multiplication (MSM)
pub fn compute_msm_baseline(point_vec: &[G1Affine], scalar_vec: &[BigInt]) -> G1Projective {
    msm::VariableBaseMSM::msm(
        point_vec,
        &scalar_vec
            .into_iter()
            .map(|x| ScalarField::from_bigint(*x).unwrap())
            .collect::<Vec<_>>(),
    )
}

/// Locally optimized version of the variable base MSM algorithm.
pub fn compute_msm<const COMPLETE: bool, const BATCH_ACC_BUCKETS: bool>(
    point_vec: &[G1Affine],
    scalar_vec: &[BigInt],
    c: Option<usize>,
) -> G1Projective {
    msm::MultiExp::compute_msm_opt::<COMPLETE, BATCH_ACC_BUCKETS>(point_vec, scalar_vec, c)
}

/// Load input vectors from the filesystem if they exist in the given directory.
/// If not, generate and save new input vectors of the requests size.
#[cfg(feature = "std")]
pub fn read_or_generate_instances<P: AsRef<Path>>(
    path: P,
    count: usize,
    size: usize,
) -> Result<Vec<Instance>, Error> {
    // Read instances from the files system and return them if available.
    match read_instances(&path) {
        Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::NotFound => (),
        result => return result,
    };

    // Generate and write the new instances to the intended directory.
    let generated: Vec<_> = (0..count).map(|_| Instance::generate(size)).collect();
    write_instances(&path, &generated, false)?;

    Ok(generated)
}

#[cfg(feature = "std")]
pub fn write_instances<P: AsRef<Path>>(
    path: P,
    instances: &[Instance],
    append: bool,
) -> Result<(), Error> {
    // If the target directory does not exist, create it.
    match path.as_ref().parent() {
        Some(dir) => create_dir_all(dir)?,
        None => (),
    };

    // If append is true, open in append mode. Otherwise truncate.
    let file = if append {
        File::options().append(true).create(true).open(path)?
    } else {
        File::create(path)?
    };

    // We use unchecked because this is not an adversarial environment and it is way faster.
    instances.serialize_unchecked(&file)?;
    Ok(())
}

#[cfg(feature = "std")]
pub fn read_instances<P: AsRef<Path>>(path: P) -> Result<Vec<Instance>, Error> {
    let file = File::open(path)?;

    // We use unchecked because this is not an adversarial environment and it is way faster.
    let instances = Vec::<Instance>::deserialize_unchecked(&file)?;
    Ok(instances)
}

#[cfg(feature = "std")]
pub fn hash<E: CanonicalSerialize>(elements: &[E]) -> Result<Hash, Error> {
    let mut buffer = vec![].writer();
    elements.serialize_unchecked(&mut buffer)?;
    Ok(blake3::hash(&buffer.into_inner()))
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_std::time::Instant;
    use serial_test::serial;
    use std::path::PathBuf;

    // Input sizes to use in the tests below.
    const K: usize = 14;
    const SIZE: usize = 1 << K;
    const TEST_DIR_BASE: &'static str = "./.test";

    fn test_instance_path(k: usize) -> PathBuf {
        Path::new(TEST_DIR_BASE)
            .join(format!("1x{}", k))
            .join("instances")
    }

    #[test]
    #[serial]
    fn baseline_msm_doesnt_panic() -> Result<(), Error> {
        let instances = read_or_generate_instances(&test_instance_path(K), 1, SIZE)?;
        let start = Instant::now();
        let res = instances[0].compute_msm_baseline();
        let duration = start.elapsed();
        println!("baseline with SIZE 1<<{}: {:?}", K, duration);
        println!("\n baseline res = {:?}\n", res.into_affine());
        Ok(())
    }

    #[test]
    #[serial]
    fn optimized_msm_doesnt_panic() -> Result<(), Error> {
        let instances = read_or_generate_instances(&test_instance_path(K), 1, SIZE)?;
        let start = Instant::now();
        let res = instances[0].compute_msm::<true, true>();
        let duration = start.elapsed();
        println!("msm_opt with SIZE 1<<{}: {:?}", K, duration);
        println!("\n msm_opt = {:?}\n", res.into_affine());
        Ok(())
    }

    #[test]
    #[serial]
    fn optimized_and_baseline_agree() -> Result<(), Error> {
        let instances = read_or_generate_instances(&test_instance_path(K), 1, SIZE)?;
        let res_base = instances[0].compute_msm_baseline();
        let res_opt = instances[0].compute_msm::<true, true>();
        assert_eq!(res_base, res_opt);
        Ok(())
    }

    #[test]
    fn serialization_derserialization_are_consistent() -> Result<(), Error> {
        let serialize_hash = {
            let instances = vec![Instance::generate(1 << 6)];
            write_instances(&test_instance_path(6), &instances, false)?;
            hash(&instances)?
        };

        let deserialize_hash = {
            let instances = read_instances(&test_instance_path(6))?;
            hash(&instances)?
        };
        assert_eq!(serialize_hash, deserialize_hash);
        Ok(())
    }
}
