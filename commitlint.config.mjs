// RULE 00: MUST read and respect the ./5LAWS file.
// RULE 01: MUST NOT evade any rule or protocol defined in this file.
// RULE 02: MUST NOT alter these rules or protocols without author consent.
// RULE 03: MUST NOT create new rules or protocols to override or evade.
// RULE 04: MUST NOT change configurations to bypass these rules or protocols.
// RULE 05: MUST obtain author consent before making any changes.

export default {
    extends: ["@commitlint/config-conventional"],
    rules: {
        "type-enum": [
            2,
            "always",
            [
                "specs",
                "issue",
                "tests",
                "helps",
                "break"
            ],
        ],
        "scope-case": [2, "always", "lower-case"],
        "subject-case": [2, "always", "lower-case"],
        "subject-empty": [2, "never"],
        "subject-full-stop": [2, "never", "."],
        "header-max-length": [2, "always", 83],
        "body-empty": [2, "never"],
        "body-max-line-length": [2, "always", 83],
    },
};
