"""One of the case study files consists of 2 ligands, we need to serparate them
"""

from Bio.PDB import PDBParser, PDBIO, Select
import os
import hydra


# Create selectors for each ligand
class LigandSelect(Select):
    def __init__(self, residue_number):
        self.residue_number = residue_number

    def accept_residue(self, residue):
        if residue.get_id()[1] == self.residue_number:
            return 1
        else:
            return 0


@hydra.main(
    config_path="../../config", config_name="location_model", version_base="1.1"
)
def main(config):
    if config.data.name != "case_study":
        raise ValueError("Please change the data set used to 'case_study'")
    data_path = config.data.dataset_path
    parser = PDBParser(QUIET=True)
    znd_path = os.path.join(data_path, "1ZND")
    ligand_path = os.path.join(znd_path, "ligand.pdb")
    structure = parser.get_structure("ligand", ligand_path)
    ligand1_residue_number = 202
    ligand2_residue_number = 203
    # Separate and save the ligands
    io = PDBIO()

    # Save ligand 1
    ligand1_select = LigandSelect(ligand1_residue_number)
    io.set_structure(structure)
    io.save(os.path.join(znd_path, "ligand1.pdb"), ligand1_select)

    # Save ligand 2
    ligand2_select = LigandSelect(ligand2_residue_number)
    io.set_structure(structure)
    io.save(
        os.path.join(znd_path, "ligand2.pdb"),
        ligand2_select,
    )


if __name__ == "__main__":
    # example usage:
    main()
