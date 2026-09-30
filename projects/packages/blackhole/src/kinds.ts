/** 与 `ton-types::IntegratorKind` `repr(u8)` 对齐。 */
export const IntegratorKind = {
    SchwarzschildOrbit: 0,
    SemiClassicalRayMarch: 1,
    Geodesic3d: 2,
    KerrEquatorial: 3,
} as const;

export type IntegratorKind = (typeof IntegratorKind)[keyof typeof IntegratorKind];

/** 与 `ton-types::DiskModelKind` `repr(u8)` 对齐。 */
export const DiskModelKind = {
    None: 0,
    NovikovThorneThin: 1,
    ProceduralFbm: 2,
} as const;

export type DiskModelKind = (typeof DiskModelKind)[keyof typeof DiskModelKind];

export const INTEGRATOR_OPTIONS: { value: IntegratorKind; label: string }[] = [
    { value: IntegratorKind.SchwarzschildOrbit, label: "史瓦西轨道" },
    { value: IntegratorKind.SemiClassicalRayMarch, label: "半经典 Ray March" },
    { value: IntegratorKind.Geodesic3d, label: "3D 测地线（未实现）" },
    { value: IntegratorKind.KerrEquatorial, label: "克尔赤道（未实现）" },
];

export const DISK_MODEL_OPTIONS: { value: DiskModelKind; label: string }[] = [
    { value: DiskModelKind.None, label: "无盘" },
    { value: DiskModelKind.NovikovThorneThin, label: "Novikov-Thorne 薄盘" },
    { value: DiskModelKind.ProceduralFbm, label: "程序化 FBM" },
];
