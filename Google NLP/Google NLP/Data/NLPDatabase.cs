using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;

namespace GoogleNLPConsole.Data
{
    public class NLPDatabase
    {
        public static List<BoardOrder> GetBoardOrders(string path = "")
        {
            var boardOrders = new List<BoardOrder>();

            var files = Directory.EnumerateFiles(path);

            foreach (var file in files)
            {
                var order = new BoardOrder()
                {
                    Path = file,
                    Text = File.ReadAllText(file),
                };
                boardOrders.Add(order);
            }

            return boardOrders;
        }

        public static void SaveBoardOrderText(string connString, string path = "")
        {
            var orders = GetBoardOrders(path);

            using (var conn = new SqlConnection(connString))
            {
                conn.Open();
                foreach (var order in orders)
                {
                    var query = $@"INSERT INTO [dbo].[BoardOrders] (FilePath, FileText,) VALUES ( @path, @text)";

                    var command = new SqlCommand(query, conn);
                    command.Parameters.AddWithValue("@path", order.Path);
                    command.Parameters.AddWithValue("@text", order.Text);

                    command.ExecuteNonQuery();
                }
                conn.Close();
            }
        }

        public static List<BoardOrder> GetBoardOrdersText(string connString)
        {
            var boardOrders = new List<BoardOrder>();

            var query = $@"SELECT FilePath, FileText, FROM [dbo].[BoardOrders]";

            using (var conn = new SqlConnection(connString))
            {
                var command = new SqlCommand(query, conn);
                conn.Open();
                using (var reader = command.ExecuteReader())
                {
                    if (reader.HasRows)
                    {
                        while (reader.Read())
                        {
                            var order = new BoardOrder()
                            {
                                Path = reader.GetString(reader.GetOrdinal("FilePath")),
                                Text = reader.GetString(reader.GetOrdinal("FileText")),
                            };
                            boardOrders.Add(order);
                        }
                    }
                }
                conn.Close();
            }

            return boardOrders;
        }

        public static void SaveResponses(List<BoardOrder> responses, string connString)
        {
            using (var conn = new SqlConnection(connString))
            {
                conn.Open();
                foreach (var response in responses)
                {
                    var query = $@"INSERT INTO [dbo].[Google] " +
                        " (Categories, DocumentSentiment, Entities, Sentences, Tokens, FilePath, Response) VALUES" +
                        " (@cat, @doc, @ent, @sen, @tok, @path, @res)";

                    var command = new SqlCommand(query, conn);
                    command.Parameters.AddWithValue("@cat", response.Categories);
                    command.Parameters.AddWithValue("@doc", response.DocumentSentiment);
                    command.Parameters.AddWithValue("@ent", response.Entities);
                    command.Parameters.AddWithValue("@sen", response.Sentences);
                    command.Parameters.AddWithValue("@tok", response.Token);
                    command.Parameters.AddWithValue("@path", response.Path);
                    command.Parameters.AddWithValue("@res", response.ResponseJson);

                    command.ExecuteNonQuery();
                }
                conn.Close();
            }
        }

        public static List<BoardOrder> GetResponses(string connString, string type)
        {
            var orderResponses = new List<BoardOrder>();

            using (var conn = new SqlConnection(connString))
            {

                var query = $@"SELECT Response, FilePath FROM [dbo].[Google] WHERE Response LIKE '%{type}%'";

                var command = new SqlCommand(query, conn);

                conn.Open();
                using (var reader = command.ExecuteReader())
                {
                    if (reader.HasRows)
                    {
                        while (reader.Read())
                        {
                            var order = new BoardOrder(reader.GetString(reader.GetOrdinal("Response")))
                            {
                                Path = reader.GetString(reader.GetOrdinal("FilePath")),
                            };

                            orderResponses.Add(order);
                        }
                    }
                }
                conn.Close();
            }

            return orderResponses;
        }

        public static List<BoardOrder> GetResponses(string connString)
        {
            var orderResponses = new List<BoardOrder>();

            using (var conn = new SqlConnection(connString))
            {

                var query = $@"SELECT Response, FilePath FROM [dbo].[Google]";

                var command = new SqlCommand(query, conn);

                conn.Open();
                using (var reader = command.ExecuteReader())
                {
                    if (reader.HasRows)
                    {
                        while (reader.Read())
                        {
                            var order = new BoardOrder(reader.GetString(reader.GetOrdinal("Response")))
                            {
                                Path = reader.GetString(reader.GetOrdinal("FilePath")),
                            };

                            orderResponses.Add(order);
                        }
                    }
                }
                conn.Close();
            }

            return orderResponses;
        }
    }
}

