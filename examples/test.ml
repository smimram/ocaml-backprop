open Backprop
open Backpropagatable

let () =
  Printf.printf "Testing...";
  let x = ref 1. in
  (* Define the sin x function. *)
  let f () =
    var x
    (* |> observe (Printf.printf "value is %f\n%!") *)
    (* |> observe_descent (Printf.printf "gradient is %f\n%!") *)
    |> sin
  in
  for _ = 0 to 100 do
    (* Evaluate the result of the function. *)
    let y = eval (f ()) in
    Printf.printf "value: %f -> %f\n%!" !x y;
    (* Optimize x in order to minimize the function. *)
    f () |> descent 0.1 |> run
  done
