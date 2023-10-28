defmodule LoraxTest do
  use ExUnit.Case
  doctest Lorax

  test "greets the world" do
    assert Lorax.hello() == :world
  end
end
