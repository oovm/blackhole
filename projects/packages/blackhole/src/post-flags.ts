/** 与 `ton-types::post_flags` 对齐。 */
export const PostFlags = {
    Raw: 0,
    UcfToneMap: 1,
    UcfBloom: 2,
} as const;

export type PostFlagsValue = number;

export function buildPostFlags(ucfToneMap: boolean, bloom: boolean): PostFlagsValue {
    let flags = 0;
    if (ucfToneMap) {
        flags |= PostFlags.UcfToneMap;
    }
    if (bloom) {
        flags |= PostFlags.UcfBloom;
    }
    return flags;
}
