(lang dune 2.7)
(name backprop)
(source (github smimram/ocaml-backprop))
(license GPL-3.0-or-later)
(authors "Samuel Mimram <samuel.mimram@polytechnique.edu>")
(maintainers "Samuel Mimram <samuel.mimram@polytechnique.edu>")

(generate_opam_files true)

(package
 (name backprop)
 (synopsis "Monadic library to perform backpropagation")
 (description "This library provides monadic support for backpropagation, which is typically used to train neural networks.")
)
