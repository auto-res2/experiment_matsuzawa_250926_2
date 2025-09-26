
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': house finch, linnet, Carpodacus mexicanus
          '1': robin, American robin, Turdus migratorius
          '2': triceratops
          '3': green mamba
          '4': harvestman, daddy longlegs, Phalangium opilio
          '5': toucan
          '6': goose
          '7': jellyfish
          '8': nematode, nematode worm, roundworm
          '9': king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes
            camtschatica
          '10': dugong, Dugong dugon
          '11': Walker hound, Walker foxhound
          '12': Ibizan hound, Ibizan Podenco
          '13': Saluki, gazelle hound
          '14': golden retriever
          '15': Gordon setter
          '16': komondor
          '17': boxer
          '18': Tibetan mastiff
          '19': French bulldog
          '20': malamute, malemute, Alaskan malamute
          '21': dalmatian, coach dog, carriage dog
          '22': Newfoundland, Newfoundland dog
          '23': miniature poodle
          '24': white wolf, Arctic wolf, Canis lupus tundrarum
          '25': African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus
          '26': Arctic fox, white fox, Alopex lagopus
          '27': lion, king of beasts, Panthera leo
          '28': meerkat, mierkat
          '29': ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle
          '30': rhinoceros beetle
          '31': ant, emmet, pismire
          '32': black-footed ferret, ferret, Mustela nigripes
          '33': three-toed sloth, ai, Bradypus tridactylus
          '34': rock beauty, Holocanthus tricolor
          '35': aircraft carrier, carrier, flattop, attack aircraft carrier
          '36': ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin,
            dustbin, trash barrel, trash bin
          '37': barrel, cask
          '38': beer bottle
          '39': bookshop, bookstore, bookstall
          '40': cannon
          '41': carousel, carrousel, merry-go-round, roundabout, whirligig
          '42': carton
          '43': catamaran
          '44': chime, bell, gong
          '45': clog, geta, patten, sabot
          '46': cocktail shaker
          '47': combination lock
          '48': crate
          '49': cuirass
          '50': dishrag, dishcloth
          '51': dome
          '52': electric guitar
          '53': file, file cabinet, filing cabinet
          '54': fire screen, fireguard
          '55': frying pan, frypan, skillet
          '56': garbage truck, dustcart
          '57': hair slide
          '58': holster
          '59': horizontal bar, high bar
          '60': hourglass
          '61': iPod
          '62': lipstick, lip rouge
          '63': miniskirt, mini
          '64': missile
          '65': mixing bowl
          '66': oboe, hautboy, hautbois
          '67': organ, pipe organ
          '68': parallel bars, bars
          '69': pencil box, pencil case
          '70': photocopier
          '71': poncho
          '72': prayer rug, prayer mat
          '73': reel
          '74': school bus
          '75': scoreboard
          '76': slot, one-armed bandit
          '77': snorkel
          '78': solar dish, solar collector, solar furnace
          '79': spider web, spider's web
          '80': stage
          '81': tank, army tank, armored combat vehicle, armoured combat vehicle
          '82': theater curtain, theatre curtain
          '83': tile roof
          '84': tobacco shop, tobacconist shop, tobacconist
          '85': unicycle, monocycle
          '86': upright, upright piano
          '87': vase
          '88': wok
          '89': worm fence, snake fence, snake-rail fence, Virginia fence
          '90': yawl
          '91': street sign
          '92': consomme
          '93': trifle
          '94': hotdog, hot dog, red hot
          '95': orange
          '96': cliff, drop, drop-off
          '97': coral reef
          '98': bolete
          '99': ear, spike, capitulum
  splits:
  - name: train
    num_bytes: 5661216817.0
    num_examples: 50000
  - name: validation
    num_bytes: 1127476741.0
    num_examples: 10000
  - name: test
    num_bytes: 663008304.0
    num_examples: 5000
  download_size: 7433313256
  dataset_size: 7451701862.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

Output:
{
    "extracted_code": ""
}
