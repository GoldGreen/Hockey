namespace Hockey.Shared.Dto
{
    public class PlayerDto
    {
        public int Id { get; set; }

        public string Type { get; set; }

        public int[] Color { get; set; }

        public double[] Bbox { get; set; }

        public double[] Center { get; set; }
    }
}
