# Third-party notices

## MaleCNS v1.0

Source: https://male-cns.janelia.org/download/

Credit: the MaleCNS collaboration, including FlyEM (HHMI Janelia), the University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research. The release page identifies the source authors and publications.

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/ (legal code: https://creativecommons.org/licenses/by/4.0/legalcode).

Derived material: `public/anatomy/`, prepared graph/interface files in the runtime archive, and the connectome-derived model artifacts. Changes include explicit neuronal filtering, simplified transmitter signs and contact-count normalization, sampled/simplified SWC branch geometry, artificial sensory/motor interfaces, and learned model parameters. See the anatomy manifest for neuron IDs, source hashes, chunk hashes and geometric tolerances; `artifacts/dataset.json` records the input hashes. No source-author endorsement is implied. The license of these underlying data is not restricted by the absence of an original-code license.

## Application scaffold and dependencies

The frontend began from a vinext/Sites scaffold. Existing notices are preserved in `build/sites-vite-plugin.LICENSE` and `vendor/shadcn-tailwind-4.13.0.LICENSE.md`. JavaScript and Python libraries retain their respective upstream licenses; lockfiles/manifests identify dependencies. They are not vendored here except where accompanied by an existing notice.

## Original work

No general open-source license has been granted for this project's original code or documentation. Obtain the owner's permission before redistribution or reuse of that original material. This notice does not supersede any applicable third-party license.
