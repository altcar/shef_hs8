/** Species table for the 20 classes in the dataset, plus the negative class. */
export type Species = { code: string; name: string; sci: string }

export const SPECIES: Species[] = [
  { code: 'amerob', name: 'American Robin', sci: 'Turdus migratorius' },
  { code: 'barswa', name: 'Barn Swallow', sci: 'Hirundo rustica' },
  { code: 'bewwre', name: "Bewick's Wren", sci: 'Thryomanes bewickii' },
  { code: 'bncfly', name: 'Brown-crested Flycatcher', sci: 'Myiarchus tyrannulus' },
  { code: 'carwre', name: 'Carolina Wren', sci: 'Thryothorus ludovicianus' },
  { code: 'comrav', name: 'Common Raven', sci: 'Corvus corax' },
  { code: 'cubthr', name: 'Curve-billed Thrasher', sci: 'Toxostoma curvirostre' },
  { code: 'eursta', name: 'European Starling', sci: 'Sturnus vulgaris' },
  { code: 'gbwwre1', name: 'Gray-breasted Wood-Wren', sci: 'Henicorhina leucophrys' },
  { code: 'houspa', name: 'House Sparrow', sci: 'Passer domesticus' },
  { code: 'houwre', name: 'Northern House Wren', sci: 'Troglodytes aedon' },
  { code: 'mallar3', name: 'Mallard', sci: 'Anas platyrhynchos' },
  { code: 'norcar', name: 'Northern Cardinal', sci: 'Cardinalis cardinalis' },
  { code: 'redcro', name: 'Red Crossbill', sci: 'Loxia curvirostra' },
  { code: 'rewbla', name: 'Red-winged Blackbird', sci: 'Agelaius phoeniceus' },
  { code: 'rubpep1', name: 'Rufous-browed Peppershrike', sci: 'Cyclorhis gujanensis' },
  { code: 'rucspa1', name: 'Rufous-collared Sparrow', sci: 'Zonotrichia capensis' },
  { code: 'sonspa', name: 'Song Sparrow', sci: 'Melospiza melodia' },
  { code: 'spotow', name: 'Spotted Towhee', sci: 'Pipilo maculatus' },
  { code: 'swathr', name: "Swainson's Thrush", sci: 'Catharus ustulatus' },
]

export const NEGATIVE = { code: 'not_a_bird', name: 'Not a bird', sci: '' }

export const LABELS: Species[] = [...SPECIES, NEGATIVE]

export const LABEL_INDEX = new Map(LABELS.map((s, i) => [s.code, i]))

export function labelOf(code: string): Species {
  return LABELS[LABEL_INDEX.get(code) ?? LABELS.length - 1]
}
