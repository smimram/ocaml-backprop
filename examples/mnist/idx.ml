let input_int ic =
  let n = ref 0 in
  for _ = 0 to 3 do
    n := !n lsl 8 + input_byte ic
  done;
  !n

let load_labels fname =
  let ic = open_in fname in
  assert (input_byte ic = 0);
  assert (input_byte ic = 0);
  assert (input_byte ic = 8); (* kind is unsigned byte *)
  assert (input_byte ic = 1); (* dimension is 1 *)
  let len = input_int ic in
  Array.init len (fun _ -> input_byte ic)

let load_images fname =
  let ic = open_in fname in
  assert (input_byte ic = 0);
  assert (input_byte ic = 0);
  assert (input_byte ic = 8); (* kind is unsigned byte *)
  assert (input_byte ic = 3); (* dimension is 3 *)
  let len = input_int ic in
  let height = input_int ic in
  let width = input_int ic in
  Array.init len
    (fun _ ->
       Array.init height
         (fun _ ->
            Array.init width
              (fun _ ->
                 input_byte ic
              )
         )
    )
