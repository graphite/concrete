use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};

use super::super::security;

/// Additional noise generated by the keyswitch step.
pub fn variance_keyswitch<W: UnsignedInteger>(
    input_lwe_dimension: u64,          //n_big
    ks_decomposition_level_count: u64, //l(BS)
    ks_decomposition_base_log: u64,    //b(BS)
    ciphertext_modulus_log: u64,
    variance_ksk: Variance,
) -> Variance {
    assert!(ciphertext_modulus_log == W::BITS as u64);
    concrete_npe::estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms::<
        W,
        Variance,
        Variance,
        BinaryKeyKind,
    >(
        LweDimension(input_lwe_dimension as usize),
        Variance(0.0),
        variance_ksk,
        DecompositionBaseLog(ks_decomposition_base_log as usize),
        DecompositionLevelCount(ks_decomposition_level_count as usize),
    )
}

/// Compute the variance paramater for `variance_keyswitch`
pub fn variance_ksk(
    internal_ks_output_lwe_dimension: u64,
    ciphertext_modulus_log: u64,
    security_level: u64,
) -> Variance {
    let glwe_poly_size = 1;
    let glwe_dim = internal_ks_output_lwe_dimension;
    security::variance_ksk(
        glwe_poly_size,
        glwe_dim,
        ciphertext_modulus_log,
        security_level,
    )
}

/// Additional noise generated by fft computation
pub fn fft_noise<W: UnsignedInteger>(
    internal_ks_output_lwe_dimension: u64, //n_small
    glwe_polynomial_size: u64,             //N
    _glwe_dimension: u64,                  //k, unused
    br_decomposition_level_count: u64,     //l(KS)
    br_decomposition_base_log: u64,        //b(ks)
) -> Variance {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L25
    let n = internal_ks_output_lwe_dimension as f64;
    let b = 2_f64.powi(br_decomposition_base_log as i32);
    let l = br_decomposition_level_count as f64;
    let big_n = glwe_polynomial_size as f64;
    // 22 = 2 x 11, 11 = 64 -53
    let scale_margin = (1_u64 << 22) as f64;
    let res = n * (0.016089458900501813 * scale_margin * l * b * b * big_n.powf(2.188930746713708));
    Variance::from_modular_variance::<W>(res)
}

/// Final reduced noise generated by the final bootstrap step.
/// Note that it does not depends from input noise, assuming the bootstrap is successful
pub fn variance_bootstrap<W: UnsignedInteger>(
    internal_ks_output_lwe_dimension: u64, //n_small
    glwe_polynomial_size: u64,             //N
    glwe_dimension: u64,                   //k
    br_decomposition_level_count: u64,     //l(KS)
    br_decomposition_base_log: u64,        //b(ks)
    ciphertext_modulus_log: u64,
    variance_bsk: Variance,
) -> Variance {
    assert!(ciphertext_modulus_log == W::BITS as u64);
    let out_variance_pbs = concrete_npe::estimate_pbs_noise::<W, Variance, BinaryKeyKind>(
        LweDimension(internal_ks_output_lwe_dimension as usize),
        PolynomialSize(glwe_polynomial_size as usize),
        GlweDimension(glwe_dimension as usize),
        DecompositionBaseLog(br_decomposition_base_log as usize),
        DecompositionLevelCount(br_decomposition_level_count as usize),
        variance_bsk,
    );
    let additional_fft_noise = fft_noise::<W>(
        internal_ks_output_lwe_dimension,
        glwe_polynomial_size,
        glwe_dimension,
        br_decomposition_level_count,
        br_decomposition_base_log,
    );
    Variance(out_variance_pbs.get_variance() + additional_fft_noise.get_variance())
}

pub fn estimate_modulus_switching_noise_with_binary_key<W>(
    internal_ks_output_lwe_dimension: u64,
    glwe_polynomial_size: u64,
) -> Variance
where
    W: UnsignedInteger,
{
    #[allow(clippy::cast_sign_loss)]
    let nb_msb = (f64::log2(glwe_polynomial_size as f64) as usize) + 1;
    concrete_npe::estimate_modulus_switching_noise_with_binary_key::<W, Variance>(
        LweDimension(internal_ks_output_lwe_dimension as usize),
        nb_msb,
        Variance(0.0),
    )
}

pub fn maximal_noise<D, W>(
    input_variance: Variance,
    input_lwe_dimension: u64,              //n_big
    internal_ks_output_lwe_dimension: u64, //n_small
    ks_decomposition_level_count: u64,     //l(BS)
    ks_decomposition_base_log: u64,        //b(BS)
    glwe_polynomial_size: u64,             //N
    ciphertext_modulus_log: u64,           //log(q)
    security_level: u64,
) -> Variance
where
    D: DispersionParameter,
    W: UnsignedInteger,
{
    assert!(ciphertext_modulus_log == W::BITS as u64);
    let v_keyswitch = variance_keyswitch::<W>(
        input_lwe_dimension,
        ks_decomposition_level_count,
        ks_decomposition_base_log,
        ciphertext_modulus_log,
        variance_ksk(
            internal_ks_output_lwe_dimension,
            ciphertext_modulus_log,
            security_level,
        ),
    );
    let v_modulus_switch = estimate_modulus_switching_noise_with_binary_key::<W>(
        internal_ks_output_lwe_dimension,
        glwe_polynomial_size,
    );
    Variance(
        input_variance.get_variance()
            + v_keyswitch.get_variance()
            + v_modulus_switch.get_variance(),
    )
}

/// The maximal noise is attained at the end of the modulus switch.
pub fn maximal_noise_multi_sum<D, W, Ignored>(
    dispersions: &[D],
    weights_tuples: &[(W, Ignored)],
    input_lwe_dimension: u64,              //n_big
    internal_ks_output_lwe_dimension: u64, //n_small
    ks_decomposition_level_count: u64,     //l(BS)
    ks_decomposition_base_log: u64,        //b(BS)
    glwe_polynomial_size: u64,             //N
    _glwe_dimension: u64,                  //k
    _br_decomposition_level_count: u64,    //l(KS)
    _br_decomposition_base_log: u64,       //b(ks)
    ciphertext_modulus_log: u64,           //log(q)
    security_level: u64,
) -> Variance
where
    D: DispersionParameter,
    W: UnsignedInteger,
{
    assert!(ciphertext_modulus_log == W::BITS as u64);
    let v_out_multi_sum = if dispersions.is_empty() {
        let mut weights = vec![];
        for (weight, _) in weights_tuples.iter() {
            weights.push(*weight);
        }
        concrete_npe::estimate_weighted_sum_noise(dispersions, weights.as_slice())
    } else {
        Variance(0.0)
    };
    maximal_noise::<D, W>(
        v_out_multi_sum,
        input_lwe_dimension,
        internal_ks_output_lwe_dimension,
        ks_decomposition_level_count,
        ks_decomposition_base_log,
        glwe_polynomial_size,
        ciphertext_modulus_log,
        security_level,
    )
}

/// The output noise is the variance boostrap.
pub fn output_noise<D, W>(
    _input_lwe_dimension: u64,             //n_big
    internal_ks_output_lwe_dimension: u64, //n_small
    _ks_decomposition_level_count: u64,    //l(BS)
    _ks_decomposition_base_log: u64,       //b(BS)
    glwe_polynomial_size: u64,             //N
    glwe_dimension: u64,                   //k
    br_decomposition_level_count: u64,     //l(KS)
    br_decomposition_base_log: u64,        //b(ks)
    ciphertext_modulus_log: u64,
    security_level: u64,
) -> Variance
where
    D: DispersionParameter,
    W: UnsignedInteger,
{
    let variance_bsk = security::variance_bsk(
        glwe_polynomial_size,
        glwe_dimension,
        ciphertext_modulus_log,
        security_level,
    );
    variance_bootstrap::<W>(
        internal_ks_output_lwe_dimension,
        glwe_polynomial_size,
        glwe_dimension,
        br_decomposition_level_count,
        br_decomposition_base_log,
        ciphertext_modulus_log,
        variance_bsk,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_python_prototype_security_variance_keyswitch_1() {
        let golden_modular_variance = 3.260702274017557e+68;
        let input_lwe_dimension = 4096;
        let internal_ks_output_lwe_dimension = 1024;
        let ks_decomposition_level_count = 9;
        let ks_decomposition_base_log = 5;
        let ciphertext_modulus_log = 128;
        let security = 128;
        let actual = variance_keyswitch::<u128>(
            input_lwe_dimension,
            ks_decomposition_level_count,
            ks_decomposition_base_log,
            ciphertext_modulus_log,
            variance_ksk(
                internal_ks_output_lwe_dimension,
                ciphertext_modulus_log,
                security,
            ),
        )
        .get_modular_variance::<u128>();
        approx::assert_relative_eq!(actual, golden_modular_variance, max_relative = 1e-8);
    }

    #[test]
    fn golden_python_prototype_security_variance_keyswitch_2() {
        // let golden_modular_variance = 8.580795457940938e+66;
        // the full npe implements a part of the full estimation
        let golden_modular_variance = 3.941898681369209e+48; // full estimation
        let input_lwe_dimension = 2048;
        let internal_ks_output_lwe_dimension = 512;
        let ks_decomposition_level_count = 2;
        let ks_decomposition_base_log = 24;
        let ciphertext_modulus_log = 64;
        let security = 128;
        let actual = variance_keyswitch::<u64>(
            input_lwe_dimension,
            ks_decomposition_level_count,
            ks_decomposition_base_log,
            ciphertext_modulus_log,
            variance_ksk(
                internal_ks_output_lwe_dimension,
                ciphertext_modulus_log,
                security,
            ),
        )
        .get_modular_variance::<u64>();
        approx::assert_relative_eq!(actual, golden_modular_variance, max_relative = 1e-8);
    }

    #[test]
    fn golden_python_prototype_security_variance_bootstrap_1() {
        // golden value include fft correction
        let golden_modular_variance = 6.283575623979502e+30;
        let internal_ks_output_lwe_dimension = 2048;
        let glwe_polynomial_size = 4096;
        let glwe_dimension = 10;
        let br_decomposition_level_count = 2;
        let br_decomposition_base_log = 24;
        let ciphertext_modulus_log = 64;
        let security = 128;
        let variance_bsk = security::variance_bsk(
            glwe_polynomial_size,
            glwe_dimension,
            ciphertext_modulus_log,
            security,
        );
        let actual = variance_bootstrap::<u64>(
            internal_ks_output_lwe_dimension,
            glwe_polynomial_size,
            glwe_dimension,
            br_decomposition_level_count,
            br_decomposition_base_log,
            ciphertext_modulus_log,
            variance_bsk,
        )
        .get_modular_variance::<u64>();
        approx::assert_relative_eq!(actual, golden_modular_variance, max_relative = 1e-8);
    }

    #[test]
    fn golden_python_prototype_security_variance_bootstrap_2() {
        // golden value include fft correction
        let golden_modular_variance = 1.3077694369436019e+56;
        let internal_ks_output_lwe_dimension = 1024;
        let glwe_polynomial_size = 4096;
        let glwe_dimension = 16;
        let br_decomposition_level_count = 9;
        let br_decomposition_base_log = 5;
        let ciphertext_modulus_log = 128;
        let security = 128;
        let variance_bsk = security::variance_bsk(
            glwe_polynomial_size,
            glwe_dimension,
            ciphertext_modulus_log,
            security,
        );
        let actual = variance_bootstrap::<u128>(
            internal_ks_output_lwe_dimension,
            glwe_polynomial_size,
            glwe_dimension,
            br_decomposition_level_count,
            br_decomposition_base_log,
            ciphertext_modulus_log,
            variance_bsk,
        )
        .get_modular_variance::<u128>();
        approx::assert_relative_eq!(actual, golden_modular_variance, max_relative = 1e-8);
    }
}
