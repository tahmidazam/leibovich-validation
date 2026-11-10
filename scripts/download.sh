#!/usr/bin/env bash
# Download the public cohorts and life tables into $LP_DATA_DIR.
#
# Usage:
#   scripts/download.sh vasudev       # Vasudev cohort ccRCC tables from the cBioPortal datahub (~100 KB)
#   scripts/download.sh transorce     # TranSORCE clinical tables from the BioImage Archive (~140 KB)
#   scripts/download.sh lifetables    # ONS 2020-based projected mortality rates (qx) (~400 KB)
#   scripts/download.sh who [ISO3...] # WHO GHE life-table rates (nMx) for the Vasudev cohort countries (~50 KB)
#   scripts/download.sh stratos       # STRATOS competing-risks example data at a pinned commit (~24 KB)
#   scripts/download.sh all           # all of the above
# Images are not needed, so the TranSORCE H&E and immunofluorescence folders are skipped.
#
# Each downloaded file is checked against data/manifest.json: cBioPortal and ONS can replace
# files at the same URL, so a changed SHA-256 stops the script. Files the manifest does not
# know yet are reported as new.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -z "${LP_DATA_DIR:-}" && -f .env ]]; then
  set -a; source .env; set +a
fi
: "${LP_DATA_DIR:?Set LP_DATA_DIR (see .env.example)}"
DATA_DIR="${LP_DATA_DIR/#\~/$HOME}"
DATA_DIR="${DATA_DIR%/}"
MANIFEST=data/manifest.json

CBIOPORTAL=https://datahub.assets.cbioportal.org
SORCE="https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/453/S-BIAD2453/Files/SORCE%20cohort"
ONS_QX=https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/mortalityratesqxprincipalprojectionunitedkingdom
# STRATOS competing-risks validation guidance (van Geloven et al., BMJ 2022,
# doi:10.1136/bmj-2021-069249), companion repo pinned at commit ba1d9e8.
STRATOS=https://raw.githubusercontent.com/survival-lumc/ValidationCompRisks/ba1d9e8e68ed4f031e55dab473ce9084694bdec9
# WHO Global Health Observatory OData API: indicator LIFE_0000000029 is the GHE
# life-table nMx (age-specific death rate) by country, sex and year.
WHO_NMX=https://ghoapi.azureedge.net/api/LIFE_0000000029
# The Vasudev cohort countries as ISO3 codes; the cBioPortal names map in
# lifetables.COUNTRY_TO_ISO3 (Czech Republic, United Kingdom, Russia, Romania,
# Serbia).
WHO_COUNTRIES=(CZE GBR ROU RUS SRB)

[[ -f "$MANIFEST" ]] || echo "warning: $MANIFEST not found; every file is reported as new" >&2

fetch() {  # url destination
  mkdir -p "$(dirname "$2")"
  curl --fail --location --silent --show-error --retry 3 -o "$2" "$1"
  echo "  fetched $(basename "$2")"
}

expected_sha() {  # path relative to DATA_DIR -> SHA-256 recorded in the manifest, or empty
  [[ -f "$MANIFEST" ]] || return 0
  awk -v key="  \"$1\": {" '
    index($0, key) == 1 {found = 1; next}
    found && /"sha256"/ {gsub(/[",]/, "", $2); print $2; exit}
  ' "$MANIFEST"
}

verify() {  # absolute path of a downloaded file under DATA_DIR
  local path="$1" rel expected actual
  rel="${path#"$DATA_DIR"/}"
  expected="$(expected_sha "$rel")"
  actual="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [[ -z "$expected" ]]; then
    echo "  NEW $rel (not in $MANIFEST)"
  elif [[ "$actual" != "$expected" ]]; then
    {
      echo "CHANGED $rel"
      echo "  the upstream file differs from $MANIFEST"
      echo "  expected $expected"
      echo "  got      $actual"
      echo "  if the replacement is deliberate, refresh the manifest stage"
    } >&2
    exit 1
  else
    echo "  OK $rel"
  fi
}

vasudev() {
  local tarball files f
  echo "vasudev: clinical tables from the cBioPortal datahub"
  tarball="$(mktemp)"
  fetch "$CBIOPORTAL/ccrcc_sjuh_2023.tar.gz" "$tarball"
  mkdir -p "$DATA_DIR/vasudev"
  # Keep only the clinical tables and curation notes; mutation calls are not used.
  tar -xzf "$tarball" -C "$DATA_DIR/vasudev" --strip-components=1 \
    ccrcc_sjuh_2023/data_clinical_patient.txt ccrcc_sjuh_2023/data_clinical_sample.txt \
    ccrcc_sjuh_2023/meta_study.txt ccrcc_sjuh_2023/readme.md
  rm -f "$tarball"
  files=(data_clinical_patient.txt data_clinical_sample.txt meta_study.txt readme.md)
  for f in "${!files[@]}"; do
    echo "  [$((f + 1))/${#files[@]}] vasudev/${files[$f]}"
    verify "$DATA_DIR/vasudev/${files[$f]}"
  done
}

transorce() {
  local names=("SORCE%20Clinical%20Data.xlsx" "SORCE%20Analysis%20Data.xlsx")
  local files=(clinical.xlsx analysis.xlsx)
  local i
  echo "transorce: clinical tables from the BioImage Archive"
  for i in "${!files[@]}"; do
    echo "  [$((i + 1))/${#files[@]}] transorce/${files[$i]}"
    fetch "$SORCE/${names[$i]}" "$DATA_DIR/transorce/${files[$i]}"
    verify "$DATA_DIR/transorce/${files[$i]}"
  done
}

lifetables() {
  # The preprint used the 2020-based projections.
  local urls=("$ONS_QX/2020based/ukppp20qx.xlsx")
  local files=(ons-2020-based-qx.xlsx)
  local i
  echo "lifetables: ONS projected mortality rates"
  for i in "${!files[@]}"; do
    echo "  [$((i + 1))/${#files[@]}] lifetables/${files[$i]}"
    fetch "${urls[$i]}" "$DATA_DIR/lifetables/${files[$i]}"
    verify "$DATA_DIR/lifetables/${files[$i]}"
  done
}

fetch_gho() {  # odata $filter expression, destination
  mkdir -p "$(dirname "$2")"
  curl --fail --location --silent --show-error --retry 3 -G \
    --data-urlencode "\$filter=$1" -o "$2" "$WHO_NMX"
  echo "  fetched $(basename "$2")"
}

who_lifetables() {
  # One OData response per country and sex, saved raw: the manifest hashes
  # them, so a changed API answer cannot drift the analysis. nMx is the
  # probability basis the WHO reader converts to single-year qx.
  local iso sex_code sex dest
  echo "who: WHO GHE life-table rates from the Global Health Observatory"
  for iso in "$@"; do
    for sex_code in SEX_MLE SEX_FMLE; do
      case "$sex_code" in
        SEX_MLE) sex=male ;;
        *) sex=female ;;
      esac
      dest="$DATA_DIR/lifetables/who/nmx-${iso}-${sex}.json"
      fetch_gho "SpatialDim eq '${iso}' and Dim1 eq '${sex_code}'" "$dest"
      verify "$dest"
    done
  done
}

stratos() {
  local files=(rdata.rds vdata.rds)
  local i
  echo "stratos: example data from the STRATOS competing-risks guidance"
  for i in "${!files[@]}"; do
    echo "  [$((i + 1))/${#files[@]}] stratos/${files[$i]}"
    fetch "$STRATOS/Data/${files[$i]}" "$DATA_DIR/stratos/${files[$i]}"
    verify "$DATA_DIR/stratos/${files[$i]}"
  done
}

case "${1:-}" in
  vasudev) vasudev ;;
  transorce) transorce ;;
  lifetables) lifetables ;;
  who) shift; [[ $# -gt 0 ]] || set -- "${WHO_COUNTRIES[@]}"; who_lifetables "$@" ;;
  stratos) stratos ;;
  all) vasudev; transorce; lifetables; who_lifetables "${WHO_COUNTRIES[@]}"; stratos ;;
  *) echo "usage: $0 {vasudev|transorce|lifetables|who [ISO3...]|stratos|all}" >&2; exit 2 ;;
esac
