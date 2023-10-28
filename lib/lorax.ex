defmodule Lorax do
  @moduledoc """
  Documentation for `Lorax`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> Lorax.hello()
      :world

  """
  def hello do
    :world
  end

  def foo(%Axon{} = axon) do
    axon
  end
end
