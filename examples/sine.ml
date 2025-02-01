open Backprop

let () =
  Printf.printf "Testing...";
  let x = ref 1. in
  (* Define the sin x function. *)
  let f () =
    Net.var x
    (* |> observe (Printf.printf "value is %f\n%!") *)
    (* |> observe_descent (Printf.printf "gradient is %f\n%!") *)
    |> Net.sin
  in
  for _ = 0 to 100 do
    (* Evaluate the result of the function. *)
    let y = Net.eval (f ()) in
    Printf.printf "value: %f -> %f\n%!" !x y;
    (* Optimize x in order to minimize the function. *)
    f () |> Net.descent 0.1 |> Net.run
  done;
  (* Minimum is reached at -π/2. *)
  assert (-1.58 <= !x && !x <= -1.57)
