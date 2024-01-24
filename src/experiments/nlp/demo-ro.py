# python -m spacy download en_core_web_sm
import spacy
from nltk import Tree
from spacy import displacy

# Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("ro_core_news_md")

# Process whole documents
text = ("Rezerva păstrată de Anglia și refuzul său de a se asocia la propunerile memoriului nu putea decât să întărească pe Poartă în acest mod de a vedea. "
        "Și dacă e adevărat, cum se asigură, că, Francia și Italia, contrariu noutăților telegrafice, n-au dat încă adeziunea lor, este vădit că Poarta s-ar afla o dată mai mult în prezența unor pasuri contradictorii din partea Europei și că nu era ușor a nu ține samă de aceasta."
        "Rămâne să știm dacă puterile Nordului, după ce au făcut să se laude prin toate foile lor oficioase soliditatea înțelegem și voința hotărâtă de a face să se esecute programa lor din punt în punt, mai cu samă după ce s-au înaintat până a decide să trămită, o escadră combinată, ca pentru a uni amenințarea cu injuncțiunea, vor suferi fără să murmure noua nereușită ce le impune răspunsul îndrăzneț al Sublimei Porți.")
doc = nlp(text)

# Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
# for entity in doc.ents:
#     print(entity.text, entity.label_)

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]            

# displacy.serve(doc, style="dep")
displacy.serve(doc, style="ent")
