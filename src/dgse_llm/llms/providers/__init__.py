from dgse_llm.registry import import_submodules

# Importing every provider module registers it, so adding a provider needs no edit here.
import_submodules(__name__)
