using Google.Cloud.Language.V1;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;

namespace GoogleNLPConsole.Data
{
    public class BoardOrder
    {
        public BoardOrder() { }
        public BoardOrder( string responseJson )
        {
            Response = JsonConvert.DeserializeObject<AnnotateTextResponse>(responseJson);
            SetResponse(Response);
        }

        public AnnotateTextResponse Response { get; set; }
        public string ResponseJson { get; set; }

        public string Path { get; set; }
        public string Text { get; set; }
        public string Categories { get; set; }
        public string DocumentSentiment { get; set; }
        public string Entities { get; set; }
        public string Sentences { get; set; }
        public string Token { get; set; }

        public void SetResponse(AnnotateTextResponse response)
        {
            Categories = response.Categories != null ? JsonConvert.SerializeObject(response.Categories, Formatting.None) : null;
            DocumentSentiment = response.DocumentSentiment != null ? JsonConvert.SerializeObject(response.DocumentSentiment, Formatting.None) : null;
            Entities = response.Entities != null ? JsonConvert.SerializeObject(response.Entities, Formatting.None) : null;
            Sentences = response.Sentences != null ? JsonConvert.SerializeObject(response.Sentences, Formatting.None) : null;
            Token = response.Tokens != null ? JsonConvert.SerializeObject(response.Tokens, Formatting.None) : null;
            Response = response.Clone();
            ResponseJson = response != null ? JsonConvert.SerializeObject(Response, Formatting.None) : null;
        }
    }

    public static class Extensions
    {
        public static IEnumerable<string> SplitByLength(this string str, int maxLength)
        {
            for (int index = 0; index < str.Length; index += maxLength)
            {
                yield return str.Substring(index, Math.Min(maxLength, str.Length - index));
            }
        }
    }
}
