use super::*;

#[test]
fn zero_round_driver_writes_mpmct_and_preserves_views() {
    let source = CircuitSeq {
        gates: vec![[0, 1, 2], [2, 0, 1]],
    };
    for (
        gadgetize,
        feistalize,
        slice_ccnot,
        five_carrier,
        strong_five_carrier,
        six_carrier,
        strong_six_carrier,
        seven_carrier,
    ) in [
        (true, false, false, false, false, false, false, false),
        (false, true, false, false, false, false, false, false),
        (true, false, true, false, false, false, false, false),
        (true, false, false, true, false, false, false, false),
        (true, false, true, true, false, false, false, false),
        (true, false, false, false, true, false, false, false),
        (true, false, true, false, true, false, false, false),
        (true, false, false, false, false, true, false, false),
        (true, false, true, false, false, true, false, false),
        (true, false, false, false, false, false, true, false),
        (true, false, true, false, false, false, true, false),
        (true, false, false, false, false, false, false, true),
        (true, false, true, false, false, false, false, true),
    ] {
        let dir = std::env::temp_dir().join(format!(
            "local_mixing_cnot_driver_{}_{}_{}_{}_{}_{}_{}_{}_{}",
            std::process::id(),
            gadgetize as u8,
            feistalize as u8,
            slice_ccnot as u8,
            five_carrier as u8,
            strong_five_carrier as u8,
            six_carrier as u8,
            strong_six_carrier as u8,
            seven_carrier as u8
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("out.txt");
        let gadget = dir.join("gadget.txt");
        let prod = if seven_carrier {
            ProdConfig::production_seven_carrier()
        } else if six_carrier || strong_six_carrier {
            ProdConfig::production_six_carrier()
        } else if five_carrier || strong_five_carrier {
            ProdConfig::production_five_carrier()
        } else {
            ProdConfig::off()
        };
        let expected_wires = if feistalize {
            9
        } else if seven_carrier {
            7 * 3 + prod.band_size(3)
        } else if six_carrier || strong_six_carrier {
            6 * 3 + prod.band_size(3)
        } else if five_carrier || strong_five_carrier {
            5 * 3 + prod.band_size(3)
        } else {
            6
        };
        let params = CnotSssParams {
            rounds: 0,
            n: 3,
            m: 1,
            x: 2,
            save: output.to_str().unwrap(),
            source: "source.txt",
            do_gadgetize: gadgetize,
            five_carrier,
            strong_five_carrier,
            six_carrier,
            strong_six_carrier,
            seven_carrier,
            do_feistalize: feistalize,
            slice_zero: false,
            slice_zero_random: false,
            slice_zero_random_gates: 96,
            slice_zero_hardcoded: false,
            slice_zero_hardcoded_rounds: 1,
            slice_zero_ccnot: slice_ccnot,
            slice_zero_ccnot_gates: if seven_carrier {
                72
            } else if six_carrier || strong_six_carrier {
                63
            } else if five_carrier || strong_five_carrier {
                54
            } else {
                18
            },
            sliced_sandwich: false,
            sandwich_balanced: false,
            sandwich_m: 12,
            sandwich_s: 6,
            gadget_path: Some(gadget.to_str().unwrap()),
            full_shuffle: false,
            full_shuffle_early: false,
            shooting_times: 1,
            collision_rounds: 1,
            stable_compressions: 1,
            expansion_game: false,
            equality_check: true,
            rg_freq: 2,
            masks: MaskConfig::off(),
            prod,
        };
        main_shuffle_shoot_shuffle_cnot(&source, &params);
        let (written, wires) = format::read_mpmct(output.to_str().unwrap()).unwrap();
        assert_eq!(wires, expected_wires);
        assert!(!written.is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
