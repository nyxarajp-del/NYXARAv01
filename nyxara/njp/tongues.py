"""NYXARA · njp/tongues.py — ordinary sentences of three real languages (🗣, NJP V.35).

:class:`~nyxara.njp.discourse.ClosedClassLearner` was measured in V.31 on **minted** languages: a
criterion fitted where the answer is known and applied to a language drawn after the lesson. That
is the right experiment and it has a gap a minted language cannot close — a drawn dialect has the
distribution its generator gave it, and the claim being made is about languages people speak.

So here are sentences of three of them. They are ordinary, short, and written out rather than
generated from a template, because the whole point is that the distribution is a language's own.
The **closed class of each is not given to her**: it sits beside the corpus as the answer key the
measurement is scored against, exactly as English's does, and nothing in
:mod:`nyxara.njp.discourse` reads it.

**What this is honestly worth.** These are small corpora — a few dozen sentences each — written by
one author, and a few dozen sentences constrain far less than a real one would. What they can show
is that the criterion is not an artefact of how :mod:`nyxara.njp.dialects` draws a language; what
they cannot show is how it behaves on a corpus of any size. Both halves are stated in the report.

**Chinese is not here, and the reason is not reluctance.** It is written without spaces between
words, and every organ in this package tokenises on them: a closed class cannot be recovered from
a language whose words this repository cannot find. Segmenting it is a real piece of work and
pretending otherwise by scoring characters would be a number with nothing behind it.

Pure data. No code that decides anything.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["SENTENCES", "CLOSED", "LANGUAGES"]

#: Ordinary sentences, by language.
SENTENCES: Dict[str, Tuple[str, ...]] = {
    "es": (
        "el perro corre en el parque",
        "la casa es muy grande",
        "los niños comen pan con queso",
        "no tengo tiempo para eso",
        "donde esta el libro rojo",
        "mi hermana vive en la ciudad",
        "el gato duerme sobre la mesa",
        "las flores del jardin son blancas",
        "yo trabajo en una oficina pequena",
        "el tren llega a las ocho",
        "ella no quiere salir hoy",
        "compramos fruta en el mercado",
        "el rio pasa cerca del pueblo",
        "los alumnos leen un libro nuevo",
        "hace frio en el invierno",
        "mi padre conduce un coche viejo",
        "la puerta de la cocina esta abierta",
        "que hora es ahora",
        "el medico habla con el paciente",
        "los pajaros cantan por la manana",
        "no entiendo esta pregunta",
        "la profesora explica la leccion",
        "el pan esta sobre la mesa",
        "vamos al cine con los amigos",
        "el nino juega en la calle",
        "la nieve cubre las montanas",
        "quien vive en esa casa",
        "el barco cruza el mar",
        "las cartas llegan por la tarde",
        "mi madre prepara la cena",
        "el reloj de la torre suena",
        "no hay agua en el pozo",
        "los obreros construyen un puente",
        "la ventana da al patio",
        "el viento mueve las hojas",
        "cuando empieza la fiesta",
        "el perro sigue al gato",
        "la ciudad tiene un museo antiguo",
    ),
    "fr": (
        "le chien court dans le parc",
        "la maison est tres grande",
        "les enfants mangent du pain",
        "je n ai pas le temps",
        "ou est le livre rouge",
        "ma soeur habite dans la ville",
        "le chat dort sur la table",
        "les fleurs du jardin sont blanches",
        "je travaille dans un bureau",
        "le train arrive a huit heures",
        "elle ne veut pas sortir",
        "nous achetons des fruits au marche",
        "la riviere passe pres du village",
        "les eleves lisent un livre",
        "il fait froid en hiver",
        "mon pere conduit une vieille voiture",
        "la porte de la cuisine est ouverte",
        "quelle heure est il",
        "le medecin parle avec le patient",
        "les oiseaux chantent le matin",
        "je ne comprends pas cette question",
        "la maitresse explique la lecon",
        "le pain est sur la table",
        "nous allons au cinema avec des amis",
        "le garcon joue dans la rue",
        "la neige couvre les montagnes",
        "qui habite dans cette maison",
        "le bateau traverse la mer",
        "les lettres arrivent le soir",
        "ma mere prepare le diner",
        "l horloge de la tour sonne",
        "il n y a pas d eau",
        "les ouvriers construisent un pont",
        "la fenetre donne sur la cour",
        "le vent bouge les feuilles",
        "quand commence la fete",
        "le chien suit le chat",
        "la ville a un vieux musee",
    ),
    "hi": (
        "कुत्ता पार्क में दौड़ता है",
        "घर बहुत बड़ा है",
        "बच्चे रोटी खाते हैं",
        "मेरे पास समय नहीं है",
        "किताब कहाँ है",
        "मेरी बहन शहर में रहती है",
        "बिल्ली मेज़ पर सोती है",
        "बगीचे के फूल सफेद हैं",
        "मैं दफ़्तर में काम करता हूँ",
        "रेल आठ बजे आती है",
        "वह आज बाहर नहीं जाना चाहती",
        "हम बाज़ार से फल खरीदते हैं",
        "नदी गाँव के पास से बहती है",
        "छात्र नई किताब पढ़ते हैं",
        "सर्दियों में ठंड होती है",
        "मेरे पिता पुरानी गाड़ी चलाते हैं",
        "रसोई का दरवाज़ा खुला है",
        "अभी क्या समय है",
        "डॉक्टर मरीज़ से बात करता है",
        "सुबह पक्षी गाते हैं",
        "मैं यह सवाल नहीं समझता",
        "अध्यापिका पाठ समझाती है",
        "रोटी मेज़ पर रखी है",
        "हम दोस्तों के साथ सिनेमा जाते हैं",
        "लड़का सड़क पर खेलता है",
        "बर्फ पहाड़ों को ढकती है",
        "उस घर में कौन रहता है",
        "जहाज़ समुद्र पार करता है",
        "चिट्ठियाँ शाम को आती हैं",
        "मेरी माँ खाना बनाती है",
        "मीनार की घड़ी बजती है",
        "कुएँ में पानी नहीं है",
        "मज़दूर पुल बनाते हैं",
        "खिड़की आँगन की ओर खुलती है",
        "हवा पत्तों को हिलाती है",
        "उत्सव कब शुरू होता है",
        "कुत्ता बिल्ली के पीछे जाता है",
        "शहर में एक पुराना संग्रहालय है",
    ),
}

#: The answer key each language is scored against. **Not given to the learner** — it sits here so
#: a measurement has something to be right or wrong about, exactly as English's shipped closed
#: class does in :mod:`nyxara.njp.semantics`.
CLOSED: Dict[str, Tuple[str, ...]] = {
    "es": ("el", "la", "los", "las", "un", "una", "de", "del", "en", "a", "al", "que", "no",
           "y", "con", "por", "para", "es", "esta", "son", "sobre", "se", "mi", "su",
           "donde", "quien", "cuando", "hay", "esa", "esto", "eso", "muy"),
    "fr": ("le", "la", "les", "un", "une", "de", "du", "des", "en", "a", "au", "que", "ne",
           "pas", "et", "dans", "sur", "avec", "pour", "est", "sont", "il", "elle", "je",
           "nous", "ma", "mon", "ou", "qui", "quand", "cette", "tres", "y", "d", "l"),
    "hi": ("में", "पर", "से", "को", "का", "की", "के", "है", "हैं", "नहीं", "और", "यह", "वह",
           "उस", "क्या", "कौन", "कहाँ", "कब", "मैं", "हम", "मेरे", "मेरी", "एक", "बहुत",
           "पास", "साथ", "ओर", "होती", "होता"),
}

#: The languages this file carries, in the order the report prints them.
LANGUAGES: Tuple[str, ...] = ("es", "fr", "hi")
