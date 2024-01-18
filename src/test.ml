open Backprop

let () =
  Printf.printf "Testing...";
  (* Printf.printf "%f\n%!" (Stdlib.sin 1.); *)
  let x = ref 1. in
  let f () =
    var 0.1 x
    (* |> observe (Printf.printf "value is %f\n%!") *)
    (* |> observe_descent (Printf.printf "gradient is %f\n%!") *)
    |> sin
  in
  for _ = 0 to 100 do
    Printf.printf "value: %f -> %f\n%!" !x (eval (f ()));
    descent (f ())
  done
