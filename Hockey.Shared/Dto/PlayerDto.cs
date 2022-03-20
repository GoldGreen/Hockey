namespace Hockey.Shared.Dto
{
    public class PlayerDto
    {
        public int Id { get; set; }

        public string Type { get; set; }
        public string Team { get; set; }

        public double[] Bbox { get; set; }
        public double[] FieldCoordinate { get; set; }
    }
}
