from re import L
import numpy as np

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
from rdkit.Chem import Draw



def plot_fig(s:str,l:int,bool_AtomIndices:bool=False):
    mol = Chem.MolFromSmiles(s)
    d = rdMolDraw2D.MolDraw2DCairo(1050, 1000)
    # mol.GetAtomWithIdx(2).SetProp('atomNote', 'foo')
    # mol.GetBondWithIdx(0).SetProp('bondNote', 'bar')
    d.drawOptions().addStereoAnnotation = False
    d.drawOptions().addAtomIndices = bool_AtomIndices
    d.drawOptions().minFontSize = 35
    d.DrawMolecule(mol)
    d.FinishDrawing()
    if bool_AtomIndices:
        extra = '_AInd'
    else:
        extra = ''
    d.WriteDrawingText(f'molecules/smile{l}{extra}_dummyatoms.png') 


def main():
    all_molecules = ["C=C/C=C/C=C/C=C/C=C/C=C",
                'C1=CC=CC=C1',
                'C12=CC=CC3=C1C(C(C=C3)=CC=C4)=C4C=C2',
                'C1(C(C=CC=C2)=C2C=C3)=C3C=CC=C1',
                'C12=CC=CC=C1C=C3C(C=CC=C3)=C2',
                'C12=C3C4=C5C6=C1C7=CC=C6C=CC5=CC=C4C=CC3=CC=C2C=C7',
                'C1(/C=C/C2=CC=CC=C2)=CC=CC=C1',
                 'C12=CC=CC=C1C=C3C(C=C(C=CC=C4)C4=C3)=C2']

    all_molecules_s = ["*=*/*=*/*=*/*=*/*=*/*=*",
                '*1=**=**=*1',
                'C12=**=CC3=C1C(C(*=*3)=**=*4)=C4*=*2',
                'C1(C(*=**=*2)=C2*=*3)=C3*=**=*1',
                'C12=**=**=C1*=C3C(*=**=*3)=*2',
                'C12=C3C4=C5C6=C1*7=**=C6*=*C5=**=C4*=*C3=**=C2*=*7',
                'C1(/*=*/C2=**=**=*2)=**=**=*1',
                 'C12=**=**=C1*=C3C(*=C(*=**=*4)C4=*3)=*2']
    for l, s in enumerate(all_molecules_s):
        plot_fig(s,l+1)
        # plot_fig(s,l+1,True)



if __name__ == "__main__":
    main()