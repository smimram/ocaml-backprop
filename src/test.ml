open Backprop

let () =
  Printf.printf "test...";
  let x = ref 0.5 in
  let f =
    let* x = var 0.1 x in
    sin x
  in
  for _ = 0 to 10 do
    Printf.printf "value: %f\n%!" (eval f);
    descent f
  done
